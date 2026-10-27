#!/usr/bin/env python3
"""Compare raw 8-bit noninterlaced RGB/RGBA PNG exports without dependencies."""
import argparse
import json
import math
from pathlib import Path
import struct
import zlib


def read_png(path):
    data = path.read_bytes()
    if data[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError(f'Not a PNG: {path}')
    offset, compressed, header = 8, bytearray(), None
    ended = False
    while offset + 12 <= len(data):
        length = struct.unpack_from('>I', data, offset)[0]
        kind = data[offset+4:offset+8]
        value = data[offset+8:offset+8+length]
        crc = struct.unpack_from('>I', data, offset+8+length)[0]
        if zlib.crc32(kind+value) & 0xffffffff != crc:
            raise ValueError('PNG CRC mismatch')
        if kind == b'IHDR': header = struct.unpack('>IIBBBBB', value)
        if kind == b'IDAT': compressed.extend(value)
        if kind == b'IEND': ended = True; break
        offset += length + 12
    if not ended or header is None:
        raise ValueError('Incomplete PNG')
    width, height, depth, color, compression, filtering, interlace = header
    if depth != 8 or color not in (2, 6) or compression or filtering or interlace:
        raise ValueError(f'Unsupported PNG encoding: {header}')
    channels = 3 if color == 2 else 4
    stride = width * channels
    raw = zlib.decompress(compressed)
    if len(raw) != height * (stride+1): raise ValueError('Invalid scanline length')
    rgb, previous = bytearray(), bytearray(stride)
    for y in range(height):
        start = y * (stride+1)
        method, row = raw[start], bytearray(raw[start+1:start+1+stride])
        if method > 4: raise ValueError('Invalid PNG filter')
        if method:
            for i in range(stride):
                left = row[i-channels] if i >= channels else 0
                above = previous[i]
                corner = previous[i-channels] if i >= channels else 0
                if method == 1: predictor = left
                elif method == 2: predictor = above
                elif method == 3: predictor = (left+above)//2
                else:
                    p = left+above-corner
                    a, b, c = abs(p-left), abs(p-above), abs(p-corner)
                    predictor = left if a <= b and a <= c else above if b <= c else corner
                row[i] = (row[i]+predictor) & 255
        if channels == 3: rgb.extend(row)
        else:
            for x in range(width): rgb.extend(row[x*4:x*4+3])
        previous = row
    return width, height, rgb


def chunk(kind, value):
    return struct.pack('>I', len(value))+kind+value+struct.pack('>I', zlib.crc32(kind+value)&0xffffffff)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--actual', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--visual', type=Path)
    args = p.parse_args()
    width, height, reference = read_png(args.reference)
    w, h, actual = read_png(args.actual)
    if (w, h) != (width, height): raise ValueError('Image dimensions differ')
    histogram = [0]*256
    for a, b in zip(reference, actual): histogram[abs(a-b)] += 1
    count = len(reference)
    mae = sum(i*n for i, n in enumerate(histogram))/count
    mse = sum(i*i*n for i, n in enumerate(histogram))/count
    report = dict(reference=str(args.reference), actual=str(args.actual), width=w, height=h,
                  rgb_mae_255=mae, rgb_rmse_255=math.sqrt(mse),
                  rgb_psnr_db=10*math.log10(255*255/mse) if mse else None,
                  max_channel_error=max(i for i, n in enumerate(histogram) if n),
                  fraction_channels_error_over_2=sum(histogram[3:])/count)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    if args.visual:
        raw = bytearray()
        for y in range(h):
            raw.append(0)
            raw.extend(reference[y*w*3:(y+1)*w*3])
            raw.extend(actual[y*w*3:(y+1)*w*3])
        args.visual.write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR', struct.pack('>IIBBBBB',w*2,h,8,2,0,0,0))
                                +chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b''))


if __name__ == '__main__': main()
