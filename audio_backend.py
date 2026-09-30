"""FMOD FSB5 inspection and PCM decoding through the bundled vgmstream CLI.

Layout checked against vgmstream src/meta/fsb5.c (r2117) and real game banks.
No FMOD SDK, game process or network connection is needed at runtime.
"""
import io
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import wave

RATES=(4000,8000,11000,11025,16000,22050,24000,32000,44100,48000,96000)
CODECS={1:'PCM8',2:'PCM16',3:'PCM24',4:'PCM32',5:'PCM float',6:'GC ADPCM',
        7:'IMA ADPCM',8:'VAG',9:'HEVAG',10:'XMA',11:'MPEG',12:'CELT',
        13:'ATRAC9',14:'XWMA',15:'Vorbis',16:'FMOD ADPCM',17:'Opus'}


def _block(raw, offset):
    if offset+28>len(raw):raise ValueError('FSB5 文件头被截断')
    version,count,headers,names,data,codec=struct.unpack_from('<6I',raw,offset+4)
    if version not in (0,1):raise ValueError('未知 FSB5 版本 %d'%version)
    base=60 if version==1 else 64
    size=base+headers+names+data
    if offset+size>len(raw) or count>100000 or headers<count*8 or (names and names<count*4):
        raise ValueError('FSB5 区块长度或音轨数无效')
    end_headers=offset+base+headers
    name_table=end_headers
    end_names=name_table+names
    cursor=offset+base;tracks=[]
    for index in range(count):
        if cursor+8>end_headers:raise ValueError('FSB5 音轨头越界')
        bits=struct.unpack_from('<Q',raw,cursor)[0];cursor+=8
        frames=bits>>34;channels=(1,2,6,8)[(bits>>5)&3]
        rate_index=(bits>>1)&15
        rate=RATES[rate_index] if rate_index<len(RATES) else 0
        data_offset=((bits>>7)&0x7FFFFFF)<<5
        extra=bits&1
        while extra:
            if cursor+4>end_headers:raise ValueError('FSB5 扩展头被截断')
            value=struct.unpack_from('<I',raw,cursor)[0];cursor+=4
            length=(value>>1)&0xFFFFFF;kind=value>>25;extra=value&1
            if cursor+length>end_headers:raise ValueError('FSB5 扩展数据越界')
            if kind==1:
                if length<1:raise ValueError('FSB5 声道信息不完整')
                channels=raw[cursor]
            if kind==2:
                if length<4:raise ValueError('FSB5 采样率信息不完整')
                rate=struct.unpack_from('<I',raw,cursor)[0]
            cursor+=length
        if not 0<channels<=64 or not 0<rate<=384000 or data_offset>data:
            raise ValueError('FSB5 音轨参数无效')
        name='音轨 %d'%(index+1)
        if names:
            relative=struct.unpack_from('<I',raw,name_table+index*4)[0]
            start=name_table+relative
            if not count*4<=relative<names:raise ValueError('FSB5 音轨名地址越界')
            end=raw.find(b'\0',start,end_names)
            if end<0:raise ValueError('FSB5 音轨名缺少结束符')
            text=raw[start:end]
            try:name=text.decode('utf8')
            except UnicodeDecodeError:name=text.decode('gb18030',errors='replace')
        tracks.append(dict(name=name or '音轨 %d'%(index+1),subsong=index+1,
            offset=offset,block_size=size,codec=CODECS.get(codec,'编码 %d'%codec),
            rate=rate,channels=channels,frames=frames,duration=frames/rate))
    return dict(offset=offset,size=size,version=version,count=count,
                names_size=names,data_size=data,codec=CODECS.get(codec,str(codec))),tracks


def inspect_bank(raw):
    blocks=[];tracks=[];cursor=0;failures=[]
    while True:
        offset=raw.find(b'FSB5',cursor)
        if offset<0:break
        try:block,samples=_block(raw,offset)
        except ValueError as ex:
            failures.append(str(ex));cursor=offset+4;continue
        blocks.append(block)
        for sample in samples:
            sample['block']=len(blocks)
            tracks.append(sample)
        cursor=offset+block['size']
    if not blocks:
        if failures or not (raw[:4]==b'RIFF' and raw[8:12]==b'FEV '):
            raise ValueError('未找到可读取的 FSB5 音频块'+('：'+failures[0] if failures else ''))
    return dict(size=len(raw),fsb5=len(blocks),blocks=blocks,tracks=tracks,
                track_count=len(tracks),names=[t['name'] for t in tracks],warnings=failures)


def decoder_path():
    path=Path(__file__).resolve().parent/'tools'/'vgmstream'/'vgmstream-cli.exe'
    if not path.is_file():
        raise FileNotFoundError('缺少音频解码器，请保留 tools/vgmstream 文件夹')
    return path


def decode_track(raw, track, decoder=None):
    exe=Path(decoder) if decoder else decoder_path()
    if track['frames']*min(track['channels'],2)*2>256*1024*1024:
        raise ValueError('音轨解码后超过 256 MB，无法在预览中加载')
    with tempfile.TemporaryDirectory(prefix='rs_fmod_') as directory:
        root=Path(directory);source=root/'audio.fsb';output=root/'audio.wav'
        start=track['offset'];end=start+track['block_size']
        if not 0<=start<end<=len(raw):raise ValueError('音频块范围无效')
        source.write_bytes(raw[start:end])
        result=subprocess.run([str(exe),'-i','-s',str(track['subsong']),'-D','2','-W','1',
                               '-o',str(output),str(source)],capture_output=True,
                              timeout=120,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        if result.returncode or not output.exists():
            detail=(result.stderr or result.stdout).decode('utf8',errors='replace')
            raise ValueError('音轨解码失败：'+detail[-500:])
        if output.stat().st_size>256*1024*1024:raise ValueError('解码输出过大')
        pcm=output.read_bytes()
        with wave.open(io.BytesIO(pcm),'rb') as wav:
            if wav.getsampwidth()!=2 or wav.getnchannels() not in (1,2):
                raise ValueError('解码器未返回可播放的 PCM16 WAV')
            if wav.getnframes()!=track['frames'] or wav.getframerate()!=track['rate']:
                raise ValueError('解码音轨长度或采样率与索引不一致')
        return pcm
