#!/usr/bin/env python3
"""Package the verified file-AAR delivery without machine-specific build files."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DELIVERY = ROOT/'engine-delivery-rknn'
ARCHIVE = ROOT/'ai-filter-enhancer-rknn-sdk-1.1.0.zip'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def main():
    meta = json.loads((DELIVERY/'MODEL_METADATA.json').read_text())
    model = DELIVERY/meta['model']
    if model.stat().st_size != meta['model_bytes'] or sha256(model) != meta['model_sha256']:
        raise RuntimeError('Model identity verification failed')
    required = ['ai-filter-enhancer-rknn-1.1.0.aar','README.md','INTEGRATION.md','VALIDATION.md',
                'MODEL_METADATA.json','consumer-example-minified.apk']
    if not all((DELIVERY/n).is_file() for n in required): raise RuntimeError('Delivery incomplete')
    files = []
    for path in sorted(DELIVERY.rglob('*')):
        relative = path.relative_to(DELIVERY)
        if not path.is_file() or any(v in ('build','.gradle','.idea') for v in relative.parts): continue
        if path.name in ('local.properties','SHA256SUMS') or path.name.startswith('.'): continue
        files.append(path)
    manifest = DELIVERY/'SHA256SUMS'
    manifest.write_text(''.join(f'{sha256(p)}  {p.relative_to(DELIVERY).as_posix()}\n' for p in files))
    files.append(manifest)
    temporary = ARCHIVE.with_suffix('.zip.partial')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in files: archive.write(path, 'engine-delivery-rknn/'+path.relative_to(DELIVERY).as_posix())
    with zipfile.ZipFile(temporary) as archive:
        bad = archive.testzip()
        if bad: raise RuntimeError(f'ZIP integrity failure: {bad}')
    temporary.replace(ARCHIVE)
    checksum = sha256(ARCHIVE)
    ARCHIVE.with_suffix('.zip.sha256').write_text(f'{checksum}  {ARCHIVE.name}\n')
    print(json.dumps({'archive':str(ARCHIVE),'bytes':ARCHIVE.stat().st_size,'sha256':checksum,'files':len(files)},indent=2))


if __name__ == '__main__': main()
