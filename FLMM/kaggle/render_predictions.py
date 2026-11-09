"""Render measured prediction/GT comparisons from downloaded evaluator artifacts."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

parser = argparse.ArgumentParser()
parser.add_argument('result_dir', type=Path)
args = parser.parse_args()
root = args.result_dir
manifest = json.loads((root/'sample-manifest.json').read_text())
rows = []
font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 16)
for info in manifest[:10]:
    idx=info['dataset_index']
    artifact=root/f'sample-{idx}.npz'
    if not artifact.exists():
        continue
    masks=np.load(artifact)
    pred=masks['prediction'].astype(bool)
    gt=masks['ground_truth'].astype(bool)
    image_file=root/'inputs'/f'{idx}.jpg'
    input_file=root/'inputs'/f'{idx}.json'
    if not image_file.exists():
        image_file=root/'input.jpg'
        input_file=root/'input.json'
    data=json.loads(input_file.read_text())
    image=Image.open(image_file).convert('RGB')
    # First target per image; selected by position, never by prediction quality.
    p,g=pred[0],gt[0]
    union=np.logical_or(p,g).sum()
    iou=float(np.logical_and(p,g).sum()/union) if union else 1.0
    panels=[]
    for mask,color,title in [(None,None,'Input'),(g,(40,190,90),'Ground truth'),(p,(245,130,35),'F-LMM prediction')]:
        arr=np.array(image).copy()
        if mask is not None:
            arr[mask]=(arr[mask]*0.5+np.array(color)*0.5).astype('uint8')
        panel=Image.fromarray(arr)
        panel.thumbnail((320,240))
        panels.append((panel,title))
    canvas=Image.new('RGB',(990,310),'white')
    draw=ImageDraw.Draw(canvas)
    query=data.get('text',[''])[0]
    draw.text((10,8),f"Image {info['img_id']} | first target IoU {iou:.3f} | {query[:90]}",fill='black',font=font)
    for j,(panel,title) in enumerate(panels):
        draw.text((10+330*j,38),title,fill='black',font=font)
        canvas.paste(panel,(10+330*j,65))
    rows.append(canvas)
if not rows:
    raise RuntimeError('No prediction masks found')
sheet=Image.new('RGB',(990,310*len(rows)), '#eeeeee')
for i,row in enumerate(rows): sheet.paste(row,(0,310*i))
sheet.save(root/'prediction-comparison.jpg',quality=92)
print(root/'prediction-comparison.jpg')
