import re, itertools
import numpy as np
import uuid
import tempfile
import os
import requests
from pypinyin import lazy_pinyin, Style
from moviepy.editor import concatenate_videoclips, ImageClip, AudioFileClip
from PIL import Image
from fastapi import APIRouter, FastAPI, HTTPException, Request
from virtual.shcemas import GenerateVideoRequest, GenerateVideoResponse
from pathlib import Path
import gc

router = APIRouter(
    prefix="/virtual",
    tags=["virtual"]
)

VIRTUAL_VIDEOS_DIR = Path("uploads") / "aividfromppt" / "videos"
VIRTUAL_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
# ----------  口型表 ----------
VIS_MAP = {
    # 中文声母 & 英文首字母 → 口型编号
    'b':'00','p':'00','m':'00',
    'f':'01',
    's':'02','x':'02','c':'02','z':'02','sh':'02','ch':'02','q':'02','zh':'02',
    'd':'03','t':'03','n':'03','l':'03','ɜ':'03','ə':'03',
    'a':'04','ɑ':'04','æ':'04',
    'A':'05',
    'i':'06','j':'06','y':'06',
    'ɔ':'07','o':'07',
    'u':'08','ʊ':'08',
    'ü':'09','v':'09',            
    # 英文辅音首字母
    'B':'00','P':'00','M':'00',
    'F':'01','V':'01',
    'S':'02','Z':'02','C':'02','ʃ':'02','tʃ':'02','dʒ':'02',
    'D':'03','T':'03','N':'03','L':'03','R':'03',
    'A':'04','E':'06','I':'06','O':'07','U':'08'
}

def phone2vis(p): 
    return VIS_MAP.get(p, '03')

def split_zh_en(text):
    return re.findall(r'([\u4e00-\u9fa5]+|[a-zA-Z]+)', text)

def tok2vis(token):
    if re.search(r'[\u4e00-\u9fa5]', token):
        return [phone2vis(py[0]) for py in lazy_pinyin(token, Style.NORMAL)]
    else:
        return [phone2vis(token[0].upper())]

def build_vis_seq(sentence):
    tokens = split_zh_en(sentence)
    return list(itertools.chain(*[tok2vis(t) for t in tokens]))

# ---------- 混合 & 视频 ----------
def blend_pair(img_a: str, img_b: str, duration: float, fps: int, blend_n: int):
    """
    混合两张口型图片，生成平滑过渡的视频片段。
    修复: 确保numpy数组在ImageClip使用期间保持有效
    """
    # 验证图片文件是否存在
    if not Path(img_a).exists():
        raise FileNotFoundError(f"口型图片不存在: {img_a}")
    if not Path(img_b).exists():
        raise FileNotFoundError(f"口型图片不存在: {img_b}")
    
    total_frames = int(duration * fps)
    still_n = max(0, total_frames - blend_n)
    clips = []
    
    try:
        # 加载图片并转换为RGBA
        img_a_pil = Image.open(img_a).convert('RGBA')
        img_b_pil = Image.open(img_b).convert('RGBA')
        
        # 生成过渡帧
        for i in range(1, blend_n + 1):
            w = i / (blend_n + 1)
            blended = Image.blend(img_a_pil, img_b_pil, w)
            # 复制数组确保数据独立，使用copy()确保内存连续
            blended_array = np.array(blended, dtype=np.uint8).copy()
            blended.close()
            clips.append(ImageClip(blended_array, duration=1/fps))
        
        # 最后一帧使用 img_b，也需要copy
        img_b_array = np.array(img_b_pil, dtype=np.uint8).copy()
        clips.append(ImageClip(img_b_array, duration=still_n/fps))
        
        # 关闭PIL图片对象
        img_a_pil.close()
        img_b_pil.close()
        
        # 连接所有片段
        result = concatenate_videoclips(clips, method="compose")
        
        return result
        
    except Exception as e:
        # 异常时清理资源
        for clip in clips:
            try:
                clip.close()
            except:
                pass
        raise Exception(f"图片混合失败: {str(e)}")

def build_smooth_video(vis_seq, fps, char_interval, blend_n, lip_dir):
    """
    根据口型序列生成平滑的视频。
    修复: 只在最终视频生成后才释放中间clips
    """
    clips = []
    
    try:
        for i, vis in enumerate(vis_seq):
            img = f"{lip_dir}/{vis}.png"
            
            # 验证图片文件是否存在
            if not Path(img).exists():
                raise FileNotFoundError(f"口型图片不存在: {img}")
                
            if i == 0:
                # 加载第一帧，使用copy确保数据独立
                img_pil = Image.open(img).convert('RGBA')
                img_array = np.array(img_pil, dtype=np.uint8).copy()
                img_pil.close()
                clips.append(ImageClip(img_array, duration=char_interval))
            else:
                prev_img = f"{lip_dir}/{vis_seq[i-1]}.png"
                clip = blend_pair(prev_img, img, char_interval, fps, blend_n)
                clips.append(clip)
        
        # 确保所有clips都已生成
        if not clips:
            raise ValueError("没有生成任何视频片段")
        
        # 连接所有片段
        result = concatenate_videoclips(clips, method="compose")
        
        return result, clips  # 返回clips以便后续清理
        
    except Exception as e:
        # 异常处理：清理资源
        for clip in clips:
            try:
                clip.close()
            except:
                pass
        raise Exception(f"拼接视频片段失败: {str(e)}")

def _load_audio_robust(audio_file_path_or_url):
    """
    稳健加载音频：支持本地路径和 http/https 远程 URL
    优化: 减小下载块大小，确保文件正确关闭
    """
    # 1. 本地文件直接加载
    if os.path.isfile(audio_file_path_or_url):
        return AudioFileClip(audio_file_path_or_url)

    # 2. 远程 URL：先下载到临时文件再加载
    if audio_file_path_or_url.startswith(("http://", "https://")):
        print(f"正在下载远程音频文件: {audio_file_path_or_url}")
        tmp_file_path = None
        try:
            response = requests.get(audio_file_path_or_url, stream=True, timeout=30)
            response.raise_for_status()

            suffix = os.path.splitext(audio_file_path_or_url)[1] or ".mp3"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file_path = tmp_file.name
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        tmp_file.write(chunk)
            
            audio = AudioFileClip(tmp_file_path)
            
            # 注册清理函数
            original_close = audio.close
            def _cleanup_close():
                original_close()
                try:
                    if tmp_file_path and os.path.exists(tmp_file_path):
                        os.unlink(tmp_file_path)
                except:
                    pass
            audio.close = _cleanup_close
            
            return audio
            
        except requests.RequestException as e:
            if tmp_file_path:
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
            raise ConnectionError(f"下载音频失败: {e}")
        except Exception as e:
            if tmp_file_path:
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
            raise

    # 3. 其他情况
    return AudioFileClip(audio_file_path_or_url)

def generate_video(text, output_video, audio_file, fps=30, char_interval=0.5, blend_n=5, gender=1):
    gender_folder = 'male' if gender == 1 else 'female'
    lip_dir = Path(__file__).parent / 'mouse-sort' / gender_folder
    
    if not lip_dir.exists():
        raise FileNotFoundError(f"口型图片目录不存在: {lip_dir}")
    
    # 生成口型序列
    vis_seq = build_vis_seq(text)
    print('viseme ->', vis_seq)
    
    video_clip = None
    audio_clip = None
    final_clip = None
    intermediate_clips = []  # 保存中间clips引用
    
    try:
        # 生成平滑视频
        video_clip, intermediate_clips = build_smooth_video(vis_seq, fps, char_interval, blend_n, str(lip_dir))
        
        # 加载音频
        audio_clip = _load_audio_robust(audio_file)
        
        # 音画对位
        final_clip = video_clip.set_audio(audio_clip)
        
        # 如果音频更长，让视频延长到音频尾
        if audio_clip.duration > video_clip.duration:
            final_clip = final_clip.set_duration(audio_clip.duration)
        
        # 输出视频文件
        final_clip.write_videofile(
            output_video,
            codec='libx264',
            audio_codec='aac',
            fps=fps,
            logger=None,
            threads=2,
            preset='medium',
            audio_bitrate='128k',
            bitrate='2000k'
        )
        
        return output_video
        
    except Exception as e:
        # 清理可能生成的不完整视频文件
        if Path(output_video).exists():
            try:
                Path(output_video).unlink()
            except:
                pass
        raise Exception(f"视频生成失败: {str(e)}")
        
    finally:
        # 1. 先关闭最终合成的clip
        if final_clip:
            try:
                final_clip.close()
            except Exception as e:
                print(f"关闭final_clip时出错: {e}")
        
        # 2. 关闭音频clip
        if audio_clip:
            try:
                audio_clip.close()
            except Exception as e:
                print(f"关闭audio_clip时出错: {e}")
        
        # 3. 关闭视频clip
        if video_clip:
            try:
                video_clip.close()
            except Exception as e:
                print(f"关闭video_clip时出错: {e}")
        
        # 4. 最后关闭所有中间clips
        for clip in intermediate_clips:
            try:
                clip.close()
            except Exception as e:
                print(f"关闭intermediate_clip时出错: {e}")
        
        intermediate_clips.clear()
        
        # 5. 强制垃圾回收
        gc.collect()


# 生成接口
@router.post(
    "/generate-video", 
    summary="生成口型视频",
    operation_id="generate_lip_sync_video",
    description="""
    生成口型同步视频。
    
    该接口根据提供的文本内容和音频文件生成口型同步的视频。
    
    参数说明：
    - text: 用于口型同步的文本内容
    - audio_file: 音频文件地址
    - gender: 说话者性别 (1 为男性, 0 为女性)
    - char_interval: 每个字符的持续时间（秒）
    
    返回生成的视频URL。
    """
)
def api_generate(req: GenerateVideoRequest):
    if not req.text:
        raise HTTPException(status_code=400, detail="文本内容不能为空")
    
    if req.gender not in [0, 1]:
        raise HTTPException(status_code=400, detail="性别参数无效，必须为 0（女性）或 1（男性）")
    
    if req.char_interval <= 0 or req.char_interval > 2:
        raise HTTPException(status_code=400, detail="字符间隔参数无效，必须在 0 到 2 秒之间")
    
    try:
        vid_name = f"{uuid.uuid4().hex}.mp4"
        save_path = VIRTUAL_VIDEOS_DIR / vid_name
        
        generate_video(
            text=req.text,
            output_video=str(save_path),
            audio_file=req.audio_file,
            gender=req.gender,
            char_interval=req.char_interval
        )
        base_url = str(Request.base_url).rstrip('/')
        relative_path = str(save_path)
        file_url = f"{base_url}/api/v1/upload/files/{relative_path}"
    
        
        # API调用结束后主动回收
        gc.collect()
                    
        return GenerateVideoResponse(
            success=True,
            video_id=vid_name.split('.')[0],
            video_url=file_url,
            message="视频生成成功"
        )
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"文件未找到: {str(e)}")
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=f"权限不足: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"视频生成失败: {str(e)}")
    finally:
        gc.collect()