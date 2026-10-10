// 构建：src.html → index.html
// 计算核心：变量改名 + 压缩（不加密字符串，保证像素循环速度）
// 界面、文案与页面水印：字符串 RC4 加密 + 改名 + 自我保护 + 反调试
const fs = require('fs');
const N = './node_modules/';
const JO = require(N + 'javascript-obfuscator');
const { minify } = require(N + 'html-minifier-terser');
const DP = process.env.DP !== '0', SD = process.env.SD !== '0';
(async () => {
  let html = fs.readFileSync('src.html', 'utf8');
  const re = /<script>\n([\s\S]*?)<\/script>/;
  const code = html.match(re)[1];
  const A = code.indexOf('/* ===================== 形态学');
  const B = code.indexOf('/* ===================== 页面与状态');
  if (A < 0 || B < 0) throw new Error('markers');
  const engineSrc = code.slice(A, B);
  let uiSrc = code.slice(0, A) + code.slice(B);
  const EN = '_0x5ad7e1';
  uiSrc = uiSrc.replace(/\bremoveWatermark\(/g, EN + '.r(').replace(/\bdetectFrame\(/g, EN + '.d(')
               .replace(/\bblankOutside\(/g, EN + '.b(').replace(/\bencodeImage\(/g, EN + '.e(').replace(/\bodd\(/g, EN + '.o(');
  const engine = JO.obfuscate(
    'var ' + EN + '=(function(){"use strict";' + engineSrc +
    '\nreturn {r:removeWatermark,d:detectFrame,b:blankOutside,e:encodeImage,o:odd};})();', {
    compact: true, target: 'browser', identifierNamesGenerator: 'hexadecimal', renameGlobals: false,
    stringArray: false, transformObjectKeys: false, controlFlowFlattening: false, deadCodeInjection: false,
    numbersToExpressions: false, simplify: true, selfDefending: false, seed: 11
  }).getObfuscatedCode();
  const ui = JO.obfuscate('(function(){' + uiSrc + '\n})();', {
    compact: true, target: 'browser', identifierNamesGenerator: 'hexadecimal', renameGlobals: false,
    stringArray: true, stringArrayEncoding: ['rc4'], stringArrayThreshold: 1,
    stringArrayRotate: true, stringArrayShuffle: true,
    stringArrayWrappersCount: 2, stringArrayWrappersType: 'function', stringArrayIndexShift: true,
    splitStrings: true, splitStringsChunkLength: 5,
    transformObjectKeys: true, selfDefending: SD,
    debugProtection: DP, debugProtectionInterval: DP ? 3000 : 0,
    disableConsoleOutput: true,
    controlFlowFlattening: true, controlFlowFlatteningThreshold: 0.5,
    deadCodeInjection: true, deadCodeInjectionThreshold: 0.2,
    simplify: true, seed: 20261009
  }).getObfuscatedCode();
  // 核心与界面两段都放进同一个 <script>，核心变量名在界面段的闭包外层
  const out = '(function(){' + engine + ui + '})();';
  html = html.replace(re, () => '<script>' + out + '</script>');
  html = await minify(html, { collapseWhitespace: true, removeComments: true, minifyCSS: true, minifyJS: false });
  fs.writeFileSync('index.html', html);
  console.log('engine', engine.length, 'ui', ui.length, 'html', html.length);
})();
