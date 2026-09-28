"""Build a five-page report from completed, auditable experiment outputs.

Run using a Python environment containing reportlab; training dependencies are
not needed. Does not train, evaluate, or alter experimental results.
"""
import json
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.graphics.shapes import Drawing, Line, String, PolyLine

ROOT = Path(__file__).resolve().parent
def read(p):
    return json.loads((ROOT/p).read_text(encoding='utf-8'))


def main():
    analysis = read('results/analysis.json')
    freeze = read('results/freeze.json')
    audit = read('results/benchmark/summary.json')
    assets = read('submission/ASSET_MANIFEST.json')
    ledger = read('results/run_ledger.json')
    runs = [(p.parent.name,json.loads(p.read_text())) for p in sorted((ROOT/'runs').glob('*/metrics.json'))
            if p.parent.name.startswith(('paired-','extended-'))]
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='TitleCustom',fontName='Helvetica-Bold',fontSize=23,leading=28,textColor=colors.HexColor('#13344A'),spaceAfter=16))
    styles.add(ParagraphStyle(name='Kicker',fontSize=9,leading=13,textColor=colors.HexColor('#167D8D'),spaceAfter=10))
    styles['BodyText'].fontSize=10
    styles['BodyText'].leading=15
    styles['BodyText'].spaceAfter=9
    styles['Heading2'].textColor=colors.HexColor('#13344A')
    styles.add(ParagraphStyle(name='SmallCustom',fontSize=8,leading=11,spaceAfter=7))
    styles.add(ParagraphStyle(name='Cell',fontSize=8,leading=11))
    story=[]
    def para(text,style='BodyText'):
        story.append(Paragraph(text,styles[style]))
    def title(number,text):
        para(f'DASE7506 / MP1 / {number}', 'Kicker')
        para(text,'TitleCustom')
    def heading(text): para(text,'Heading2')
    def table(rows,widths):
        wrapped=[[Paragraph(escape(str(c)),styles['Cell']) for c in row] for row in rows]
        t=Table(wrapped,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E3EEF3')),
                              ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),
                              ('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),
                              ('BOTTOMPADDING',(0,0),(-1,-1),7),
                              ('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#90A8B4')),
                              ('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#D6E0E5'))]))
        story.extend([t,Spacer(1,12)])

    title('01 / OVERVIEW','A controlled small-language-model study')
    para('Parameter-matched SwiGLU under a fixed training budget','Heading2')
    para('Individual coursework | WikiText-2 | Fixed BPE-2048 | September 2026','Kicker')
    heading('Abstract')
    para(escape(analysis['abstract']))
    heading('Task and experimental question')
    para('The objective is next-token prediction from random initialization using only the supplied training split. '
         'The question is whether replacing the baseline GELU feed-forward sublayer with a near-parameter-matched '
         'SwiGLU sublayer improves full-validation bits per byte (BPB) at an equal number of processed training targets. '
         'The final submitted model uses the same training-target budget as the baseline, so additional training '
         'is not responsible for the measured architectural comparison.')
    heading('Protocol and constraints')
    para('The fixed protocol is <b>7506-mp1-wt2-v2</b>: vocabulary 2,048, independent causal windows of 256 targets, '
         'including the final short window. The model has no external retrieval assets, no cross-window state, '
         'and no evaluation network access. The supplied data, tokenizer, common utilities and scorer are unchanged.')
    table([['Constraint','Required limit'],['CPU scoring time','At most 5 times the supplied baseline on the same machine'],
           ['Peak evaluation RAM','At most 4 GiB'],['Uncompressed inference assets','At most 64 MiB']], [155,350])
    heading('Metric')
    para('BPB = summed negative natural-log probability / ln(2) / raw UTF-8 byte count. Lower is better. '
         'The denominator follows the course protocol and includes the entire raw split. '
         'Validation has 376,599 scored targets and 1,148,007 bytes; test has 428,405 targets and 1,292,013 bytes.')
    story.append(PageBreak())

    title('02 / METHOD','One isolated architectural change')
    heading('Baseline and candidate')
    para('The provided GPT has four blocks, width 128, four attention heads, learned absolute positions, '
         'pre-layer normalization, causal scaled dot-product attention, and tied input/output embeddings. '
         'These existing components are retained and are not claimed as new contributions.')
    para('<b>Baseline:</b> FFN(x) = W2 GELU(W1 x + b1) + b2, with inner width 512.<br/>'
         '<b>Candidate:</b> FFN(x) = Wd [SiLU(Wg x + bg) * (Wv x + bv)] + bd, with inner width 344. '
         'The multiplication is elementwise. A fused gate/value projection is split into two tensors.')
    para('SwiGLU follows Shazeer [1]. Its input-dependent multiplicative gate may improve the use of a limited feature budget. '
         'This is a hypothesis to test on this dataset, not a guarantee. Three projections require a narrower inner dimension '
         'than the baseline two-projection FFN: 3Dh approximately equals 8D squared, so h is approximately 8D/3, '
         'rounded to a multiple of eight. Biases and rounding prevent an exactly equal parameter count.')
    heading('Controlled comparison and ablation')
    para('Each paired run uses 1,200 updates, batch size 32, context 256, and 9,830,400 processed targets. '
         'Seeds 17 and 29 are repeated across both models. The data-start generator is seeded separately, so paired runs '
         'see the same sampled training windows. Model initialization follows the supplied normal initialization; new '
         'FFN tensors necessarily differ between architectures. The gate-disabled implementation is tested to reproduce '
         'the supplied baseline exactly under an identical seed, making the baseline the mechanism-removal ablation.')
    heading('Training and checkpoint selection')
    para('The supplied recipe uses AdamW, peak learning rate 0.001, weight decay 0.1, 100-step warmup, cosine decay '
         'to 10% of peak, and gradient clipping at 1.0. Paired runs use the same training precision. '
         'Validation is evaluated in FP32 every 300 steps. The trainer now saves both the final checkpoint and the best '
         'validation checkpoint; fixed-budget comparisons use final checkpoints. Selection never uses test BPB.')
    para(escape(analysis['extension_method']))
    heading('Correctness')
    para('The original contract tests cover causal prefix independence, normalized finite probabilities, sample independence, '
         'reset between windows, finite nonzero gradients, and exact target coverage. An additional test verifies that '
         'gate removal equals the baseline. Protected files are checked against release SHA-256 hashes.')
    story.append(PageBreak())

    title('03 / VALIDATION','Results before the test set')
    rows=[['Run','Params','Targets (M)','Final val. BPB','Best val. BPB','Train (s)']]
    for name,m in runs:
        rows.append([name.replace('paired-','').replace('extended-','long-'),f"{m['parameters']:,}",
                     f"{m['train_tokens']/1e6:.2f}",f"{m['validation']['bpb']:.5f}",
                     f"{m['best_validation_bpb']:.5f}",f"{m['train_seconds']:.1f}"])
    table(rows,[133,65,65,80,80,82])
    para('All values above are measured. Best-checkpoint BPB is reported separately from the final-checkpoint equal-budget comparison. '
         'Training time is as recorded by the trainer; intermediate validation and checkpoint work can affect wall-clock cost.','SmallCustom')
    heading('Interpretation')
    for text in analysis['validation_discussion']: para(escape(text))
    # Compact vector chart, generated only from measured validation history.
    paired=[(name,m) for name,m in runs if name.startswith('paired-')]
    points=[r for _,m in paired for r in m['validation_history']]
    if points:
        chart=Drawing(505,160)
        xmin,xmax=0,1200
        ymin=min(p['bpb'] for p in points)-.03
        ymax=max(p['bpb'] for p in points)+.03
        chart.add(Line(42,25,495,25,strokeColor=colors.grey))
        chart.add(Line(42,25,42,145,strokeColor=colors.grey))
        palette=['#21718C','#6EAAC0','#C66B3C','#D5A082']
        for k,(name,m) in enumerate(paired):
            coords=[]
            for r in m['validation_history']:
                coords += [42+453*r['step']/xmax,25+120*(r['bpb']-ymin)/(ymax-ymin)]
            if len(coords)>=4: chart.add(PolyLine(coords,strokeColor=colors.HexColor(palette[k%4]),strokeWidth=1.6))
            chart.add(String(55+(k%2)*240,153-(k//2)*12,name.replace('paired-',''),fontSize=7,fillColor=colors.HexColor(palette[k%4])))
        for step in (300,600,900,1200): chart.add(String(36+453*step/xmax,12,str(step),fontSize=7))
        chart.add(String(0,137,f'{ymax:.2f}',fontSize=7)); chart.add(String(0,25,f'{ymin:.2f}',fontSize=7))
        story.append(chart)
        para('Figure 1. Full-validation BPB by update for the paired experiments; lower is better.','SmallCustom')
    story.append(PageBreak())

    title('04 / FROZEN TEST','Reproduction and resource audit')
    para(escape(analysis['freeze_description']))
    table([['Measured quantity','Baseline','Frozen candidate'],
           ['Full-test FP32 CPU BPB',f"{audit['baseline']['bpb']:.8f}",f"{audit['candidate']['bpb']:.8f}"],
           ['Median CPU scoring time (s)',f"{audit['baseline']['median_scoring_seconds']:.3f}",f"{audit['candidate']['median_scoring_seconds']:.3f}"],
           ['Peak process RAM (GiB)',f"{audit['baseline']['max_peak_process_ram_bytes']/1024**3:.3f}",f"{audit['candidate']['max_peak_process_ram_bytes']/1024**3:.3f}"],
           ['CPU time / baseline','1.000',f"{audit['time_ratio']:.3f}"],
           ['Uncompressed inference files','-',f"{assets['total_bytes']/1024**2:.3f} MiB"]],[235,135,135])
    heading('Measurement procedure')
    para('The unchanged evaluator runs in fresh Python processes, with CPU, FP32 and four PyTorch threads. '
         'Three trials alternate baseline/candidate order. The comparison uses median scorer time, excluding loading. '
         'Both run at Below Normal process priority on Windows to reduce desktop disruption. '
         'Peak RAM covers loading and the evaluator process tree, including the Windows virtual-environment launcher '
         'and its Python child. At 20 ms intervals, psutil sums their peak-working-set counters (or current RSS if larger); '
         'the maximum sum is a conservative memory estimate. Absolute times are machine-specific. The reference is the original architecture '
         'trained at seed 17 for 1,200 steps, not a hard-coded instructor runtime.')
    para('Hardware: AMD Ryzen 7 5800H (8 cores / 16 logical processors), approximately 16 GiB physical RAM, '
         'NVIDIA GeForce RTX 3060 Laptop GPU with 6 GiB VRAM. GPU training uses PyTorch 2.7.1+cu126; '
         'CPU evaluation uses the CPU build of PyTorch 2.7.1; exact installed package versions are recorded '
         'in results/environment-cpu.txt and results/environment-gpu.txt.')
    heading('Identity and assets')
    para('The final checkpoint is <b>submission/model.pt</b>. Its configuration and implementation module are embedded in '
         'the checkpoint. Source and checkpoint hashes were recorded in results/freeze.json before scoring the test split. '
         'The asset manifest counts the checkpoint, required model source, fixed tokenizer and configuration '
         'conservatively (audit metadata itself is excluded); no learned database or hidden external file is required.')
    para('Checkpoint SHA-256: '+freeze['checkpoint_sha256'],'SmallCustom')
    heading('Cost disclosure')
    para(escape(analysis['cost_description']))
    story.append(PageBreak())

    title('05 / DISCUSSION','Limits, reproducibility and attribution')
    heading('What the evidence establishes')
    for text in analysis['limitations']: para(escape(text))
    heading('Reproduce the frozen score')
    para('Install Python 3.12, PyTorch 2.7.1 and the provided requirements. From code/, run:','BodyText')
    para('python evaluate.py --checkpoint submission/model.pt<br/>'
         '--device cpu --precision fp32 --threads 4 --split test','SmallCustom')
    para('The command is a single shell command; the line break above is for layout. Use the returned <b>bpb</b> field. '
         'No retraining is necessary. REPRODUCE.md documents training, tests and resource measurement. '
         'The source archive includes the fixed benchmark, raw experiment metrics and reproduction commands.')
    heading('AI assistance and reused work')
    para('The baseline, scorer, dataset packaging and tokenizer are supplied by the course. SwiGLU is an existing mechanism [1]. '
         'OpenAI Codex assisted with experimental design, implementation, tests, execution, analysis and drafting this report '
         'and the accompanying documentation. The student is responsible for personally reviewing and understanding the '
         'implementation before submitting; this report does not assert that this review has already occurred.')
    heading('References')
    para('[1] Noam Shazeer. GLU Variants Improve Transformer. 2020. https://arxiv.org/abs/2002.05202','SmallCustom')
    para('[2] DASE7506 MP1 course starter package. GUIDE.md, README.md, model.py and fixed evaluation protocol 7506-mp1-wt2-v2. 2026.','SmallCustom')
    para('[3] Stephen Merity et al. Pointer Sentinel Mixture Models. 2016. https://arxiv.org/abs/1609.07843. '
         'WikiText text is by Wikipedia contributors; retain the course README attribution to CC BY-SA 3.0 and GFDL.','SmallCustom')
    out=ROOT/'output/pdf/MP1_Report.pdf'
    out.parent.mkdir(parents=True,exist_ok=True)
    def footer(canvas,doc):
        canvas.setStrokeColor(colors.HexColor('#CCD9E1')); canvas.line(45,39,550,39)
        canvas.setFont('Helvetica',8); canvas.setFillColor(colors.HexColor('#58717F'))
        canvas.drawString(45,26,'DASE7506 | Small Language Model Challenge | Reproducible local experiments')
        canvas.drawRightString(550,26,str(doc.page))
    doc=SimpleDocTemplate(str(out),pagesize=(595.28,841.89),leftMargin=45,rightMargin=45,topMargin=43,bottomMargin=53,
                          title='MP1: Parameter-matched SwiGLU under a fixed training budget',author='Student submission draft with disclosed AI assistance')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print(out)


if __name__ == '__main__':
    main()
