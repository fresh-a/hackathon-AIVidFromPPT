import re, itertools
import uuid
import tempfile
import os
import requests
import subprocess
from pypinyin import lazy_pinyin, Style
from fastapi import APIRouter, FastAPI, HTTPException, Request
from virtual.shcemas import GenerateVideoRequest, GenerateVideoResponse
from pathlib import Path
import gc
import shutil

router = APIRouter(prefix="/virtual", tags=["virtual"])

VIRTUAL_VIDEOS_DIR = Path("uploads") / "aividfromppt" / "videos"
VIRTUAL_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

# ----------  口型表 ----------
VIS_MAP = {
    'b': '00',
    'p': '00',
    'm': '00',
    'f': '01',
    's': '02',
    'x': '02',
    'c': '02',
    'z': '02',
    'sh': '02',
    'ch': '02',
    'q': '02',
    'zh': '02',
    'd': '03',
    't': '03',
    'n': '03',
    'l': '03',
    'ɜ': '03',
    'ə': '03',
    'a': '04',
    'ɑ': '04',
    'æ': '04',
    'A': '05',
    'i': '06',
    'j': '06',
    'y': '06',
    'ɔ': '07',
    'o': '07',
    'u': '08',
    'ʊ': '08',
    'ü': '09',
    'v': '09',
    'B': '00',
    'P': '00',
    'M': '00',
    'F': '01',
    'V': '01',
    'S': '02',
    'Z': '02',
    'C': '02',
    'ʃ': '02',
    'tʃ': '02',
    'dʒ': '02',
    'D': '03',
    'T': '03',
    'N': '03',
    'L': '03',
    'R': '03',
    'E': '06',
    'I': '06',
    'O': '07',
    'U': '08',
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


def _load_audio_robust(audio_file_path_or_url, temp_dir):
    """加载音频文件"""
    if os.path.isfile(audio_file_path_or_url):
        return audio_file_path_or_url, False

    if audio_file_path_or_url.startswith(("http://", "https://")):
        print(f"正在下载音频: {audio_file_path_or_url}")
        try:
            response = requests.get(audio_file_path_or_url, stream=True, timeout=30)
            response.raise_for_status()

            suffix = os.path.splitext(audio_file_path_or_url)[1] or ".mp3"
            tmp_audio_path = os.path.join(temp_dir, f"audio_{uuid.uuid4().hex}{suffix}")

            with open(tmp_audio_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            return tmp_audio_path, True
        except requests.RequestException as e:
            raise ConnectionError(f"下载音频失败: {e}")

    if os.path.exists(audio_file_path_or_url):
        return audio_file_path_or_url, False
    else:
        raise FileNotFoundError(f"音频文件不存在: {audio_file_path_or_url}")


def get_audio_duration(audio_path):
    """
    获取音频时长（秒）
    推荐版本：结合多种方法，快速且可靠
    """
    import json
    import re

    #  处理可能的 tuple 输入
    if isinstance(audio_path, tuple):
        audio_path = audio_path[0]

    # 转换为字符串
    audio_path = str(audio_path)

    # 验证文件存在
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")

    # 方法1: JSON格式获取（最可靠且信息完整）
    try:
        cmd = [
            'ffprobe',
            '-v',
            'error',
            '-print_format',
            'json',
            '-show_format',
            '-show_streams',
            audio_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout)

            # 优先从 format 获取
            if 'format' in data and 'duration' in data['format']:
                duration_str = str(data['format']['duration'])
                if duration_str != 'N/A':
                    duration = float(duration_str)
                    if duration > 0:
                        print(f"✓ 音频时长: {duration:.2f}秒")
                        return duration

            # 备用：从音频流获取
            if 'streams' in data:
                for stream in data['streams']:
                    if stream.get('codec_type') == 'audio' and 'duration' in stream:
                        duration_str = str(stream['duration'])
                        if duration_str != 'N/A':
                            duration = float(duration_str)
                            if duration > 0:
                                print(f"✓ 音频时长(stream): {duration:.2f}秒")
                                return duration
    except Exception as e:
        print(f"JSON方法失败: {e}")

    # 方法2: 从ffmpeg输出解析（兜底方案，最可靠）
    try:
        print("使用ffmpeg解析音频信息...")
        cmd = ['ffmpeg', '-i', audio_path, '-f', 'null', '-']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        # 从stderr解析Duration
        duration_match = re.search(
            r'Duration:\s*(\d{2}):(\d{2}):(\d{2}\.\d{2})', result.stderr
        )
        if duration_match:
            hours = int(duration_match.group(1))
            minutes = int(duration_match.group(2))
            seconds = float(duration_match.group(3))
            duration = hours * 3600 + minutes * 60 + seconds
            if duration > 0:
                print(f"音频时长(ffmpeg): {duration:.2f}秒")
                return duration
    except Exception as e:
        print(f"ffmpeg方法失败: {e}")

    # 所有方法失败
    raise Exception(f"无法获取音频时长: {audio_path}")


def create_segment_video(
    img_a, img_b, duration, fps, blend_n, output_path, is_first=False
):
    """
    使用FFmpeg直接创建单个片段视频（包含混合效果）
    关键：使用FFmpeg的blend滤镜直接处理图片混合
    """
    try:
        if is_first:
            # 第一个片段：只显示第一张图片
            cmd = [
                'ffmpeg',
                '-y',
                '-loop',
                '1',
                '-i',
                img_a,
                '-t',
                str(duration),
                '-vf',
                f'fps={fps},format=yuv420p',
                '-c:v',
                'libx264',
                '-preset',
                'ultrafast',
                '-crf',
                '23',
                output_path,
            ]
        else:
            # 后续片段：从img_a过渡到img_b
            total_frames = int(duration * fps)
            blend_frames = min(blend_n, total_frames)
            still_frames = total_frames - blend_frames

            if blend_frames <= 0:
                # 没有混合帧，直接显示img_b
                cmd = [
                    'ffmpeg',
                    '-y',
                    '-loop',
                    '1',
                    '-i',
                    img_b,
                    '-t',
                    str(duration),
                    '-vf',
                    f'fps={fps},format=yuv420p',
                    '-c:v',
                    'libx264',
                    '-preset',
                    'ultrafast',
                    '-crf',
                    '23',
                    output_path,
                ]
            else:
                # 使用FFmpeg blend滤镜创建混合效果
                blend_duration = blend_frames / fps
                still_duration = still_frames / fps

                # 创建混合部分
                temp_blend = output_path.replace('.mp4', '_blend.mp4')

                # blend滤镜：从img_a淡入到img_b
                blend_cmd = [
                    'ffmpeg',
                    '-y',
                    '-loop',
                    '1',
                    '-t',
                    str(blend_duration),
                    '-i',
                    img_a,
                    '-loop',
                    '1',
                    '-t',
                    str(blend_duration),
                    '-i',
                    img_b,
                    '-filter_complex',
                    f'[0:v][1:v]blend=all_expr=\'A*(1-T/{blend_duration})+B*T/{blend_duration}\':shortest=1,fps={fps},format=yuv420p',
                    '-c:v',
                    'libx264',
                    '-preset',
                    'ultrafast',
                    '-crf',
                    '23',
                    temp_blend,
                ]

                result = subprocess.run(blend_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    raise Exception(f"混合视频生成失败: {result.stderr}")

                if still_frames > 0:
                    # 创建静止部分
                    temp_still = output_path.replace('.mp4', '_still.mp4')
                    still_cmd = [
                        'ffmpeg',
                        '-y',
                        '-loop',
                        '1',
                        '-i',
                        img_b,
                        '-t',
                        str(still_duration),
                        '-vf',
                        f'fps={fps},format=yuv420p',
                        '-c:v',
                        'libx264',
                        '-preset',
                        'ultrafast',
                        '-crf',
                        '23',
                        temp_still,
                    ]

                    result = subprocess.run(still_cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        raise Exception(f"静止视频生成失败: {result.stderr}")

                    # 合并混合部分和静止部分
                    concat_file = output_path.replace('.mp4', '_concat.txt')
                    with open(concat_file, 'w') as f:
                        f.write(f"file '{os.path.basename(temp_blend)}'\n")
                        f.write(f"file '{os.path.basename(temp_still)}'\n")

                    concat_cmd = [
                        'ffmpeg',
                        '-y',
                        '-f',
                        'concat',
                        '-safe',
                        '0',
                        '-i',
                        concat_file,
                        '-c',
                        'copy',
                        output_path,
                    ]

                    result = subprocess.run(
                        concat_cmd,
                        capture_output=True,
                        text=True,
                        cwd=os.path.dirname(output_path),
                    )
                    if result.returncode != 0:
                        raise Exception(f"合并视频失败: {result.stderr}")

                    # 清理临时文件
                    try:
                        os.remove(temp_blend)
                        os.remove(temp_still)
                        os.remove(concat_file)
                    except:
                        pass
                else:
                    # 只有混合部分
                    shutil.move(temp_blend, output_path)

                return

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"片段视频生成失败: {result.stderr}")

    except Exception as e:
        raise Exception(f"创建视频片段失败: {str(e)}")


def generate_video_ffmpeg_fast(
    vis_seq, fps, char_interval, blend_n, lip_dir, audio_path, output_video, temp_dir
):
    """
    极速版本：每个口型片段独立生成，然后合并
    """
    try:
        print(f"开始生成视频，共 {len(vis_seq)} 个口型片段...")

        # 创建片段目录
        segments_dir = os.path.join(temp_dir, 'segments')
        os.makedirs(segments_dir, exist_ok=True)

        segment_files = []

        # 并行生成每个片段（逐个处理，避免内存问题）
        for i, vis in enumerate(vis_seq):
            print(f"处理片段 {i+1}/{len(vis_seq)}: {vis}")

            img_current = os.path.join(lip_dir, f"{vis}.png")
            if not os.path.exists(img_current):
                raise FileNotFoundError(f"口型图片不存在: {img_current}")

            segment_output = os.path.join(segments_dir, f"segment_{i:04d}.mp4")

            if i == 0:
                # 第一个片段
                create_segment_video(
                    img_current,
                    img_current,
                    char_interval,
                    fps,
                    blend_n,
                    segment_output,
                    is_first=True,
                )
            else:
                # 后续片段
                img_prev = os.path.join(lip_dir, f"{vis_seq[i-1]}.png")
                create_segment_video(
                    img_prev,
                    img_current,
                    char_interval,
                    fps,
                    blend_n,
                    segment_output,
                    is_first=False,
                )

            segment_files.append(segment_output)

        # 合并所有片段
        print("合并所有视频片段...")
        concat_list = os.path.join(temp_dir, 'segments_concat.txt')
        with open(concat_list, 'w') as f:
            for seg_file in segment_files:
                f.write(f"file '{seg_file}'\n")

        temp_video = os.path.join(temp_dir, 'video_no_audio.mp4')
        concat_cmd = [
            'ffmpeg',
            '-y',
            '-f',
            'concat',
            '-safe',
            '0',
            '-i',
            concat_list,
            '-c',
            'copy',
            temp_video,
        ]

        result = subprocess.run(concat_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"合并视频失败: {result.stderr}")

        audio_file = audio_path[0] if isinstance(audio_path, tuple) else audio_path
        audio_duration = get_audio_duration(audio_file)

        try:
            video_duration = get_audio_duration(temp_video)
        except:
            video_duration = len(vis_seq) * char_interval

        # 如果音频更长，延长视频
        final_video = temp_video
        if audio_duration > video_duration + 0.1:  # 留一点容差
            print(
                f"延长视频以匹配音频 ({video_duration:.2f}s -> {audio_duration:.2f}s)..."
            )

            last_img = os.path.join(lip_dir, f"{vis_seq[-1]}.png")
            extra_duration = audio_duration - video_duration

            temp_extra = os.path.join(temp_dir, 'extra.mp4')
            extra_cmd = [
                'ffmpeg',
                '-y',
                '-loop',
                '1',
                '-i',
                last_img,
                '-t',
                str(extra_duration),
                '-vf',
                f'fps={fps},format=yuv420p',
                '-c:v',
                'libx264',
                '-preset',
                'ultrafast',
                '-crf',
                '23',
                temp_extra,
            ]

            subprocess.run(extra_cmd, capture_output=True, check=True)

            # 合并原视频和延长部分
            concat_final_list = os.path.join(temp_dir, 'final_concat.txt')
            with open(concat_final_list, 'w') as f:
                f.write(f"file '{temp_video}'\n")
                f.write(f"file '{temp_extra}'\n")

            temp_video_extended = os.path.join(temp_dir, 'video_extended.mp4')
            concat_final_cmd = [
                'ffmpeg',
                '-y',
                '-f',
                'concat',
                '-safe',
                '0',
                '-i',
                concat_final_list,
                '-c',
                'copy',
                temp_video_extended,
            ]

            subprocess.run(concat_final_cmd, capture_output=True, check=True)
            final_video = temp_video_extended

        # 合并音频
        print("合并音视频...")
        merge_cmd = [
            'ffmpeg',
            '-y',
            '-i',
            final_video,
            '-i',
            audio_file,
            '-c:v',
            'copy',
            '-c:a',
            'aac',
            '-b:a',
            '128k',
            '-shortest',
            output_video,
        ]

        result = subprocess.run(merge_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"音视频合并失败: {result.stderr}")

        print(f"视频生成成功: {output_video}")
        return output_video

    except Exception as e:
        raise Exception(f"视频生成失败: {str(e)}")


def generate_video(
    text, output_video, audio_file, fps=30, char_interval=0.5, blend_n=5, gender=1
):
    gender_folder = 'male' if gender == 1 else 'female'
    lip_dir = Path(__file__).parent / 'mouse-sort' / gender_folder

    if not lip_dir.exists():
        raise FileNotFoundError(f"口型图片目录不存在: {lip_dir}")

    # 生成口型序列
    vis_seq = build_vis_seq(text)
    print('口型序列 ->', vis_seq)

    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix='lipsync_')
    print(f"临时目录: {temp_dir}")

    try:
        # 加载音频
        print("处理音频...")
        audio_path = _load_audio_robust(audio_file, temp_dir)

        # 生成视频
        generate_video_ffmpeg_fast(
            vis_seq,
            fps,
            char_interval,
            blend_n,
            str(lip_dir),
            audio_path,
            output_video,
            temp_dir,
        )

        return output_video

    except Exception as e:
        if Path(output_video).exists():
            try:
                Path(output_video).unlink()
            except:
                pass
        raise Exception(f"视频生成失败: {str(e)}")

    finally:
        try:
            if os.path.exists(temp_dir):
                print(f"清理临时目录: {temp_dir}")
                shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"清理临时目录时出错: {e}")

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
    """,
)
def api_generate(req: GenerateVideoRequest, request: Request):
    if not req.text:
        raise HTTPException(status_code=400, detail="文本内容不能为空")

    if req.gender not in [0, 1]:
        raise HTTPException(
            status_code=400, detail="性别参数无效，必须为 0（女性）或 1（男性）"
        )

    if req.char_interval <= 0 or req.char_interval > 2:
        raise HTTPException(
            status_code=400, detail="字符间隔参数无效，必须在 0 到 2 秒之间"
        )

    subtitle_url = req.subtitle_url

    try:
        vid_name = f"{uuid.uuid4().hex}.mp4"
        save_path = VIRTUAL_VIDEOS_DIR / vid_name

        generate_video(
            text=req.text,
            output_video=str(save_path),
            audio_file=req.audio_file,
            gender=req.gender,
            char_interval=req.char_interval,
        )

        base_url = str(request.base_url).rstrip('/')
        relative_path = str(save_path)
        file_url = f"{base_url}/api/v1/upload/files/{relative_path}"

        gc.collect()

        return GenerateVideoResponse(
            success=True,
            video_url=file_url,
            subtitle_url=subtitle_url,
            message="视频生成成功",
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"文件未找到: {str(e)}")
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=f"权限不足: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"视频生成失败: {str(e)}")
    finally:
        gc.collect()
