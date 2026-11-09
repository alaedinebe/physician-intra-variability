const fs = require('fs');
const path = require('path');
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType, ShadingType, AlignmentType,
        PageOrientation, ImageRun, HeadingLevel, BorderStyle, PageBreak } = require('docx');
const REPO = require('path').resolve(__dirname, '..', 'results', 'manuscript');
const tables = JSON.parse(fs.readFileSync(REPO + '/tables/tables.json', 'utf8'));
tables['Table S1'] = JSON.parse(fs.readFileSync(__dirname + '/s1.json', 'utf8'));
const caps = JSON.parse(fs.readFileSync(__dirname + '/captions.json', 'utf8'));
const figs = JSON.parse(fs.readFileSync(__dirname + '/fig_captions.json', 'utf8'));
const FONT = 'Arial';
const W = 15398 - 2 * 1000; // A4 landscape width minus margins (DXA)
const thin = { style: BorderStyle.SINGLE, size: 4, color: '999999' };
const none = { style: BorderStyle.NONE, size: 0, color: 'FFFFFF' };

function para(text, opts = {}) {
  return new Paragraph({ spacing: { after: opts.after ?? 120 }, children: [new TextRun({ text, font: FONT, size: opts.size ?? 20, bold: opts.bold })] });
}
function cell(text, width, header, section) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    shading: header ? { type: ShadingType.CLEAR, fill: 'EDEDED', color: 'auto' } : undefined,
    borders: { top: header ? thin : none, bottom: thin, left: none, right: none },
    margins: { top: 40, bottom: 40, left: 70, right: 70 },
    children: [new Paragraph({ children: [new TextRun({ text: String(text), font: FONT, size: 15, bold: header || section })] })],
  });
}
function makeTable(t) {
  const n = t.columns.length;
  const first = Math.round(W * (n > 7 ? 0.16 : 0.2));
  const rest = Math.floor((W - first) / (n - 1));
  const widths = [first, ...Array(n - 1).fill(rest)];
  widths[n - 1] = W - first - rest * (n - 2);
  const rows = [new TableRow({ tableHeader: true, children: t.columns.map((c, i) => cell(c, widths[i], true)) })];
  for (const r of t.rows) {
    const section = /^[A-C]\. /.test(r[0]) && r.slice(1).every(v => v === '' || v === 'nan');
    rows.push(new TableRow({ children: r.map((v, i) => cell(v === 'nan' ? '' : v, widths[i], false, section)) }));
  }
  return new Table({ width: { size: W, type: WidthType.DXA }, columnWidths: widths, rows });
}
const order = ['Table 5', 'Table 6', 'Table 7', 'Table 8', 'Table 9', 'Table S1', 'Table S2', 'Table S3', 'Table S4',
               'Table S5', 'Table S6', 'Table S7', 'Table S8', 'Table S9'];
const children = [
  new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: 'Tables and figures of the revised manuscript', font: FONT })] }),
  para('Generated from the benchmark outputs by scripts/manuscript_tables.py and scripts/manuscript_figures.py. Tables 5 to 9 belong in the main text; Tables S1 to S9 form Additional file 1. Figures are provided separately as PNG (300 dpi) and PDF files and are reproduced at the end with their captions. Minus signs are typographic (−).', { size: 18 }),
];
for (const name of order) {
  children.push(new Paragraph({ children: [new PageBreak()] }));
  children.push(para(caps[name], { size: 18 }));
  children.push(makeTable(tables[name]));
}
for (const [file, cap] of figs) {
  children.push(new Paragraph({ children: [new PageBreak()] }));
  const png = fs.readFileSync(`${REPO}/figures/${file}.png`);
  // read PNG size
  const w = png.readUInt32BE(16), h = png.readUInt32BE(20);
  const maxW = 900, maxH = 520;
  const s = Math.min(maxW / w, maxH / h);
  children.push(new Paragraph({ alignment: AlignmentType.CENTER, children: [new ImageRun({ type: 'png', data: png, transformation: { width: Math.round(w * s), height: Math.round(h * s) } })] }));
  children.push(para(cap, { size: 18 }));
}
const doc = new Document({
  sections: [{ properties: { page: { size: { width: 11906, height: 16838, orientation: PageOrientation.LANDSCAPE }, margin: { top: 900, bottom: 900, left: 1000, right: 1000 } } }, children }],
});
Packer.toBuffer(doc).then(b => { fs.writeFileSync(process.argv[2], b); console.log('written'); });
