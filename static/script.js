/* ═══════════════════════════════════════════════════════════════
   MicroScope QC v2 — Frontend Logic
   ═══════════════════════════════════════════════════════════════ */

const $  = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const els = {
  fileInput:    $('#fileInput'),
  dropZone:     $('#dropZone'),
  uploadFrame:  $('.upload-frame'),
  uploadContent:$('#uploadContent'),
  uploadPreview:$('#uploadPreview'),
  previewImg:   $('#previewImg'),
  previewName:  $('#previewName'),
  previewDim:   $('#previewDim'),
  browseBtn:    $('#browseBtn'),
  clearBtn:     $('#clearBtn'),
  analyzeBtn:   $('#analyzeBtn'),
  hero:         $('#hero'),
  loading:      $('#loadingSection'),
  results:      $('#resultsSection'),
  newAnalysis:  $('#newAnalysisBtn'),
};

let selectedFile = null;
let currentReport = null;

// ════════════ FILE SELECTION ════════════
function handleFile(file) {
  if (!file || (file.type && !file.type.startsWith('image/'))) {
    alert('Please select a valid image file.');
    return;
  }
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    els.previewImg.src = e.target.result;
    els.previewImg.onload = () => {
      els.previewDim.textContent = `${els.previewImg.naturalWidth} × ${els.previewImg.naturalHeight} px · ${(file.size/1024).toFixed(0)} KB`;
    };
    els.previewName.textContent = file.name;
    els.uploadContent.classList.add('hidden');
    els.uploadPreview.classList.remove('hidden');
  };
  reader.readAsDataURL(file);
}

function resetUpload() {
  selectedFile = null;
  els.fileInput.value = '';
  els.uploadContent.classList.remove('hidden');
  els.uploadPreview.classList.add('hidden');
}

els.browseBtn.addEventListener('click', () => els.fileInput.click());
els.uploadFrame.addEventListener('click', (e) => {
  if (e.target.closest('.upload-preview')) return;
  els.fileInput.click();
});
els.clearBtn.addEventListener('click', (e) => { e.stopPropagation(); resetUpload(); });
els.fileInput.addEventListener('change', (e) => {
  if (e.target.files[0]) handleFile(e.target.files[0]);
});

['dragover', 'dragenter'].forEach(ev =>
  els.uploadFrame.addEventListener(ev, (e) => {
    e.preventDefault();
    els.uploadFrame.classList.add('dragging');
  })
);
['dragleave', 'drop'].forEach(ev =>
  els.uploadFrame.addEventListener(ev, (e) => {
    e.preventDefault();
    els.uploadFrame.classList.remove('dragging');
  })
);
els.uploadFrame.addEventListener('drop', (e) => {
  if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
});


// ════════════ ANALYSIS ════════════
els.analyzeBtn.addEventListener('click', async (e) => {
  e.stopPropagation();
  if (!selectedFile) return;
  await runAnalysis(selectedFile);
});

els.newAnalysis.addEventListener('click', () => {
  els.results.classList.add('hidden');
  els.hero.classList.remove('hidden');
  resetUpload();
  window.scrollTo({ top: 0, behavior: 'smooth' });
});

async function runAnalysis(file) {
  els.hero.classList.add('hidden');
  els.results.classList.add('hidden');
  els.loading.classList.remove('hidden');

  const steps = $$('.loading-step');
  steps.forEach(s => s.classList.remove('active', 'done'));
  let stepIdx = 0;
  steps[0].classList.add('active');
  const stepTimer = setInterval(() => {
    if (stepIdx < steps.length - 1) {
      steps[stepIdx].classList.remove('active');
      steps[stepIdx].classList.add('done');
      stepIdx++;
      steps[stepIdx].classList.add('active');
    }
  }, 500);

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/api/analyze', { method: 'POST', body: formData });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Analysis failed');
    }
    const data = await res.json();
    currentReport = data;
    clearInterval(stepTimer);
    steps.forEach(s => { s.classList.remove('active'); s.classList.add('done'); });
    await new Promise(r => setTimeout(r, 300));

    els.loading.classList.add('hidden');
    els.results.classList.remove('hidden');
    renderResults(data);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  } catch (err) {
    clearInterval(stepTimer);
    alert('Error: ' + err.message);
    els.loading.classList.add('hidden');
    els.hero.classList.remove('hidden');
  }
}


// ════════════ RENDERING ════════════
function scoreColor(s) {
  if (s >= 75) return '#00e5a0';
  if (s >= 45) return '#ffb800';
  return '#ff3d5a';
}

function verdictClass(decision) {
  return { PASS: 'pass', REVIEW: 'review', REJECT: 'reject' }[decision] || 'pass';
}

function renderResults(data) {
  // Score
  const score = data.overall_score;
  animateNumber($('#overallScore'), 0, score, 1200);

  // Verdict label
  const lbl = $('#scoreLabel');
  const verdictTextMap = {
    PASS:   'GOOD FOR ANALYSIS',
    REVIEW: 'MANUAL REVIEW',
    REJECT: 'NOT SUITABLE',
  };
  lbl.textContent = verdictTextMap[data.verdict.decision] || data.verdict.decision;
  lbl.className = 'score-label ' + verdictClass(data.verdict.decision);

  // Confidence
  $('#confidenceValue').textContent = `${Math.round(data.verdict.confidence * 100)}%`;

  // Arc meter
  const arc = $('#meterArc');
  const offset = 628.3 - (score / 100) * 628.3;
  arc.style.transition = 'stroke-dashoffset 1.4s cubic-bezier(.16,1,.3,1)';
  setTimeout(() => { arc.style.strokeDashoffset = offset; }, 100);

  const ticks = $('#meterTicks');
  ticks.innerHTML = '';
  for (let i = 0; i < 60; i++) {
    const angle = (i / 60) * 360 - 90;
    const rad = angle * Math.PI / 180;
    const r1 = 110, r2 = i % 5 === 0 ? 117 : 114;
    const x1 = 120 + Math.cos(rad) * r1;
    const y1 = 120 + Math.sin(rad) * r1;
    const x2 = 120 + Math.cos(rad) * r2;
    const y2 = 120 + Math.sin(rad) * r2;
    ticks.innerHTML += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="rgba(255,255,255,${i % 5 === 0 ? 0.3 : 0.1})" stroke-width="1"/>`;
  }

  // Image profile
  $('#infoDim').textContent = `${data.image_info.width} × ${data.image_info.height} px`;
  let critCount = 0, warnCount = 0;
  Object.values(data.metrics).forEach(m => {
    m.findings.forEach(f => {
      if (f.severity === 'fail') critCount++;
      else if (f.severity === 'warn') warnCount++;
    });
  });
  $('#infoCritical').textContent = critCount;
  $('#infoCritical').style.color = critCount > 0 ? 'var(--danger)' : 'var(--accent)';
  $('#infoWarnings').textContent = warnCount;
  $('#infoWarnings').style.color = warnCount > 0 ? 'var(--warn)' : 'var(--accent)';

  const recMap = {
    PASS:   ['PROCEED',       'var(--accent)'],
    REVIEW: ['MANUAL REVIEW', 'var(--warn)'],
    REJECT: ['REJECT',        'var(--danger)'],
  };
  const [recText, recColor] = recMap[data.verdict.decision] || ['—', 'var(--text)'];
  $('#infoRec').textContent = recText;
  $('#infoRec').style.color = recColor;

  $('#infoVersion').textContent = `v${data.version}`;

  // Decision trail
  renderDecisionTrail(data.verdict);

  // Metrics
  ['blur', 'lighting', 'noise', 'density'].forEach(key => {
    renderMetricCard(key, data.metrics[key]);
  });

  // Visualizations
  $('#annotatedImg').src     = data.images.annotated || data.images.original;
  $('#heatmapImg').src       = data.images.heatmap   || data.images.original;
  $('#compareOriginal').src  = data.images.original;
  $('#compareAnnotated').src = data.images.annotated || data.images.original;
  drawHistogram(data.histogram);

  // Findings ledger
  renderFindings(data);
}


function renderDecisionTrail(verdict) {
  const trail = $('#decisionTrail');
  trail.innerHTML = '';
  verdict.reasoning.forEach((step, idx) => {
    const li = document.createElement('li');
    li.className = 'decision-step';
    li.style.animation = `fade-up 0.4s ${idx * 0.1}s both ease`;
    const outcomeClass = step.outcome.toLowerCase();
    li.innerHTML = `
      <div class="ds-label">${step.step}</div>
      <div class="ds-detail">${step.detail}</div>
      <div class="ds-outcome ${outcomeClass}">${step.outcome}</div>
    `;
    trail.appendChild(li);
  });

  const blockersBlock = $('#blockersBlock');
  const blockersList = $('#blockersList');
  if (verdict.blockers && verdict.blockers.length > 0) {
    blockersBlock.classList.remove('hidden');
    blockersList.innerHTML = verdict.blockers.map(rule =>
      `<span class="blocker-tag"><code style="font-family:inherit;background:none;border:none;">${rule}</code></span>`
    ).join('');
  } else {
    blockersBlock.classList.add('hidden');
  }
}


function renderMetricCard(key, m) {
  const card = $(`#card${key.charAt(0).toUpperCase() + key.slice(1)}`);
  card.classList.remove('severity-pass', 'severity-warn', 'severity-fail');
  card.classList.add(`severity-${m.severity}`);

  animateNumber(card.querySelector('.ms-num'), 0, m.score, 1000);

  const fill = card.querySelector('.metric-fill');
  setTimeout(() => {
    fill.style.width = m.score + '%';
    fill.style.background = scoreColor(m.score);
  }, 100);

  const sev = card.querySelector('.metric-sev');
  sev.textContent = m.severity.toUpperCase();
  sev.className = `metric-sev ${m.severity}`;

  const raw = card.querySelector('.metric-raw');
  raw.innerHTML = Object.entries(m.measurements).map(([k, v]) =>
    `<div class="raw-row"><span>${k.replace(/_/g, ' ')}</span><span>${v}</span></div>`
  ).join('');

  const issues = card.querySelector('.metric-issues');
  if (m.findings.length === 0) {
    issues.innerHTML = `<div class="metric-issue pass">No findings</div>`;
  } else {
    issues.innerHTML = m.findings.map(f =>
      `<div class="metric-issue ${f.severity}">${f.message}</div>`
    ).join('');
  }
}


function renderFindings(data) {
  const list = $('#issuesList');
  const allFindings = [];
  Object.entries(data.metrics).forEach(([cat, m]) => {
    m.findings.forEach(f => {
      allFindings.push({ category: cat.toUpperCase(), ...f });
    });
  });

  // Sort: fail first, warn next, pass last
  const order = { fail: 0, warn: 1, pass: 2 };
  allFindings.sort((a, b) => order[a.severity] - order[b.severity]);

  if (allFindings.length === 0) {
    list.innerHTML = `<li class="no-issues">✓ NO FINDINGS</li>`;
    return;
  }

  list.innerHTML = allFindings.map(f => `
    <li class="issue-item" data-severity="${f.severity}">
      <span class="issue-cat ${f.severity}">${f.severity.toUpperCase()}</span>
      <span class="issue-rule">${f.rule_id}</span>
      <div class="issue-content">
        <div class="issue-message">${f.message}</div>
        <div class="issue-formula">
          <span class="measured">${f.metric}=${f.measured}</span>
          <span class="threshold">${f.operator} ${f.threshold} threshold</span>
        </div>
        <div class="issue-impact">→ ${f.impact}</div>
      </div>
    </li>
  `).join('');
}


// Findings filter
$$('.filter-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    $$('.filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const filter = btn.dataset.filter;
    $$('.issue-item').forEach(item => {
      if (filter === 'all') {
        item.classList.remove('hidden-by-filter');
      } else {
        if (item.dataset.severity === filter) {
          item.classList.remove('hidden-by-filter');
        } else {
          item.classList.add('hidden-by-filter');
        }
      }
    });
  });
});


// ════════════ HISTOGRAM ════════════
function drawHistogram(histData) {
  const svg = $('#histogramSvg');
  const W = 600, H = 240, PAD = 8;
  const colors = { Red: '#ff3d5a', Green: '#00e5a0', Blue: '#0099ff' };

  let maxVal = 0;
  Object.values(histData).forEach(arr => {
    arr.forEach(v => { if (v > maxVal) maxVal = v; });
  });

  let svgContent = '';
  for (let i = 0; i <= 4; i++) {
    const y = PAD + (H - 2*PAD) * (i / 4);
    svgContent += `<line x1="0" y1="${y}" x2="${W}" y2="${y}" stroke="#1c2330" stroke-width="0.5"/>`;
  }
  for (let i = 0; i <= 8; i++) {
    const x = (W * i / 8);
    svgContent += `<line x1="${x}" y1="${PAD}" x2="${x}" y2="${H-PAD}" stroke="#1c2330" stroke-width="0.5"/>`;
  }

  Object.entries(histData).forEach(([channel, data]) => {
    const colour = colors[channel];
    const points = data.map((v, i) => {
      const x = (i / 255) * W;
      const y = H - PAD - (v / maxVal) * (H - 2 * PAD);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    });
    const areaPath = `M 0,${H-PAD} L ` + points.join(' L ') + ` L ${W},${H-PAD} Z`;
    svgContent += `<path d="${areaPath}" fill="${colour}" opacity="0.12"/>`;
    const linePath = 'M ' + points.join(' L ');
    svgContent += `<path d="${linePath}" fill="none" stroke="${colour}" stroke-width="1.2" opacity="0.85"/>`;
  });

  svg.innerHTML = svgContent;
}


// ════════════ TABS ════════════
$$('.viz-tab').forEach(tab => {
  tab.addEventListener('click', () => {
    const target = tab.dataset.tab;
    $$('.viz-tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    $$('.viz-content').forEach(c => c.classList.remove('active'));
    $(`.viz-content[data-content="${target}"]`).classList.add('active');
  });
});


// ════════════ NUMBER ANIMATION ════════════
function animateNumber(el, from, to, duration = 1000) {
  const start = performance.now();
  const ease = (t) => 1 - Math.pow(1 - t, 3);
  function tick(now) {
    const elapsed = now - start;
    const progress = Math.min(elapsed / duration, 1);
    const value = from + (to - from) * ease(progress);
    el.textContent = Math.round(value);
    if (progress < 1) requestAnimationFrame(tick);
    else el.textContent = Math.round(to);
  }
  requestAnimationFrame(tick);
}


// ════════════ HEALTH PING ════════════
fetch('/api/health').then(r => r.json()).then(data => {
  if (data.version) {
    $('#versionTag').textContent = `v${data.version} · AI-Powered Cell Detection`;
  }
}).catch(() => {
  const status = $('#apiStatus');
  status.innerHTML = '<span class="pulse" style="background:#ff3d5a;box-shadow:0 0 8px #ff3d5a"></span> SYSTEM OFFLINE';
  status.style.color = '#ff3d5a';
  status.style.borderColor = 'rgba(255,61,90,0.25)';
  status.style.background = 'rgba(255,61,90,0.05)';
});
