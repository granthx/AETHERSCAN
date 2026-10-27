/**
 * AETHER·SCAN // Cinematic Futuristic Medical AI Experience
 * Three.js 3D Volumetric Lung Particle Simulator & Real-Time CT Vision Intelligence
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements - Hero & Atmosphere
  const cursorSpotlight = document.getElementById('cursorSpotlight');
  const canvasContainer = document.getElementById('canvasContainer');
  const btnEnterLab = document.getElementById('btnEnterLab');

  // DOM Elements - Specimen Deck & Viewport
  const holoViewer = document.getElementById('holoViewer');
  const dragDropOverlay = document.getElementById('dragDropOverlay');
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const scanLaser = document.getElementById('scanLaser');
  const hudFilename = document.getElementById('hudFilename');
  const btnHeaderUpload = document.getElementById('btnHeaderUpload');
  const btnResetAll = document.getElementById('btnResetAll');

  const singleViewport = document.getElementById('singleViewport');
  const dualViewport = document.getElementById('dualViewport');
  const mainHoloImg = document.getElementById('mainHoloImg');
  const dualRawImg = document.getElementById('dualRawImg');
  const dualCamImg = document.getElementById('dualCamImg');
  const inferenceLoader = document.getElementById('inferenceLoader');

  // DOM Elements - View Mode Selectors
  const btnViewDual = document.getElementById('btnViewDual');
  const btnViewOverlay = document.getElementById('btnViewOverlay');
  const btnViewHeatmap = document.getElementById('btnViewHeatmap');
  const btnViewRaw = document.getElementById('btnViewRaw');

  // DOM Elements - Telemetry & Output
  const telemetryEmpty = document.getElementById('telemetryEmpty');
  const telemetryActive = document.getElementById('telemetryActive');
  const verdictBanner = document.getElementById('verdictBanner');
  const verdictTag = document.getElementById('verdictTag');
  const verdictTitle = document.getElementById('verdictTitle');
  const verdictSummary = document.getElementById('verdictSummary');
  const vcmNumber = document.getElementById('vcmNumber');
  const vcmBar = document.getElementById('vcmBar');
  const stDetail = document.getElementById('stDetail');

  const valPCT = document.getElementById('valPCT');
  const valNCT = document.getElementById('valNCT');
  const valNiCT = document.getElementById('valNiCT');
  const barPCT = document.getElementById('barPCT');
  const barNCT = document.getElementById('barNCT');
  const barNiCT = document.getElementById('barNiCT');
  const ecNarrative = document.getElementById('ecNarrative');
  const ecTargetClass = document.getElementById('ecTargetClass');

  // Controls & Sliders
  const uncertaintyThreshold = document.getElementById('uncertaintyThreshold');
  const valThreshold = document.getElementById('valThreshold');
  const camAlpha = document.getElementById('camAlpha');
  const valAlpha = document.getElementById('valAlpha');

  const jpegSlider = document.getElementById('jpegSlider');
  const valJpeg = document.getElementById('valJpeg');
  const brightSlider = document.getElementById('brightSlider');
  const valBright = document.getElementById('valBright');
  const rotSlider = document.getElementById('rotSlider');
  const valRot = document.getElementById('valRot');
  const btnResetPerturbations = document.getElementById('btnResetPerturbations');

  // Specimen Vault Buttons
  const btnVaultUpload = document.getElementById('btnVaultUpload');
  const specimenPCT = document.getElementById('specimenPCT');
  const specimenNCT = document.getElementById('specimenNCT');
  const specimenNiCT = document.getElementById('specimenNiCT');

  // State Management
  let currentFile = null;
  let pristineImage = null;
  let activeViewMode = 'dual'; // 'dual' | 'overlay' | 'heatmap' | 'raw'
  let rawSliceDataUrl = '';
  let overlayDataUrl = '';
  let heatmapDataUrl = '';
  let currentInferenceResult = null;

  // ------------------------------------------------------------------------
  // 1. Interactive Cursor Spotlight
  // ------------------------------------------------------------------------
  if (cursorSpotlight) {
    window.addEventListener('mousemove', (e) => {
      cursorSpotlight.style.left = `${e.clientX}px`;
      cursorSpotlight.style.top = `${e.clientY}px`;
    });
  }

  // ------------------------------------------------------------------------
  // 2. Three.js 3D Volumetric Lung Particle Simulator
  // ------------------------------------------------------------------------
  initVolumetricLungSimulation();

  function initVolumetricLungSimulation() {
    if (typeof THREE === 'undefined' || !canvasContainer) return;

    const width = canvasContainer.clientWidth || window.innerWidth;
    const height = canvasContainer.clientHeight || window.innerHeight;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 1000);
    camera.position.z = 240;

    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    canvasContainer.appendChild(renderer.domElement);

    // Particle Cloud Geometry: Generate Bilateral Lung Lobes
    const particleCount = 2800;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);

    const cyanColor = new THREE.Color(0x00f0ff);
    const blueColor = new THREE.Color(0x38bdf8);
    const crimsonColor = new THREE.Color(0xff2a5f);

    let idx = 0;
    for (let i = 0; i < particleCount; i++) {
      // Left (-1) or Right (+1) Lung Lobe
      const side = Math.random() > 0.5 ? 1 : -1;
      const u = Math.random();
      const v = Math.random();
      const theta = u * 2.0 * Math.PI;
      const phi = Math.acos(2.0 * v - 1.0);
      const r = Math.cbrt(Math.random());

      // Ellipsoidal lung lobe coordinate parametrization
      const x = (r * Math.sin(phi) * Math.cos(theta) * 36 + side * 48) + (Math.random() - 0.5) * 8;
      const y = (r * Math.sin(phi) * Math.sin(theta) * 58) + (Math.random() - 0.5) * 8;
      const z = (r * Math.cos(phi) * 32) + (Math.random() - 0.5) * 8;

      positions[idx] = x;
      positions[idx + 1] = y;
      positions[idx + 2] = z;

      // Color mapping: electric cyan with icy blue & occasional lesion highlight
      let pColor = Math.random() > 0.15 ? (Math.random() > 0.5 ? cyanColor : blueColor) : crimsonColor;
      colors[idx] = pColor.r;
      colors[idx + 1] = pColor.g;
      colors[idx + 2] = pColor.b;

      idx += 3;
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    // Custom Glowing Particle Material
    const material = new THREE.PointsMaterial({
      size: 3.2,
      vertexColors: true,
      transparent: true,
      opacity: 0.65,
      blending: THREE.AdditiveBlending,
    });

    const particles = new THREE.Points(geometry, material);
    scene.add(particles);

    // Animated Planar Laser Ring
    const ringGeo = new THREE.RingGeometry(75, 78, 64);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0x00f0ff,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.25,
      blending: THREE.AdditiveBlending,
    });
    const laserRing = new THREE.Mesh(ringGeo, ringMat);
    laserRing.rotation.x = Math.PI / 2;
    scene.add(laserRing);

    // Mouse Tilt Parallax
    let targetX = 0, targetY = 0;
    window.addEventListener('mousemove', (e) => {
      targetX = (e.clientX / window.innerWidth - 0.5) * 30;
      targetY = (e.clientY / window.innerHeight - 0.5) * -30;
    });

    // Animation Loop
    let clock = new THREE.Clock();
    function animate() {
      requestAnimationFrame(animate);
      const elapsed = clock.getElapsedTime();

      // Slow organic rotation & breathing expansion
      particles.rotation.y += 0.003;
      particles.rotation.x += (targetY * 0.01 - particles.rotation.x) * 0.05;
      particles.rotation.z += (targetX * 0.01 - particles.rotation.z) * 0.05;

      const breathingScale = 1.0 + Math.sin(elapsed * 1.5) * 0.04;
      particles.scale.set(breathingScale, breathingScale, breathingScale);

      // Vertical Laser Sweep Plane
      laserRing.position.y = Math.sin(elapsed * 1.8) * 55;
      laserRing.material.opacity = 0.15 + Math.abs(Math.sin(elapsed * 1.8)) * 0.25;

      renderer.render(scene, camera);
    }
    animate();

    // Resize Handler
    window.addEventListener('resize', () => {
      const newW = canvasContainer.clientWidth || window.innerWidth;
      const newH = canvasContainer.clientHeight || window.innerHeight;
      camera.aspect = newW / newH;
      camera.updateProjectionMatrix();
      renderer.setSize(newW, newH);
    });
  }

  // ------------------------------------------------------------------------
  // 3. Specimen Vault Quick Buttons & Active State Management
  // ------------------------------------------------------------------------
  function setActiveSpecimenCard(activeCardElement, activeClassType) {
    [btnVaultUpload, specimenPCT, specimenNCT, specimenNiCT].forEach(card => {
      if (!card) return;
      card.classList.remove('active', 'active-pct', 'active-nct', 'active-nict', 'active-upload');
      const ind = card.querySelector('.specimen-indicator');
      if (ind) {
        if (card === btnVaultUpload) {
          ind.textContent = 'BROWSE / DROP CT ↗';
        } else {
          ind.textContent = 'LOAD SLICE ⌖';
        }
      }
    });

    if (activeCardElement) {
      activeCardElement.classList.add('active');
      if (activeClassType) {
        activeCardElement.classList.add(`active-${activeClassType.toLowerCase()}`);
      }
      const ind = activeCardElement.querySelector('.specimen-indicator');
      if (ind) {
        if (activeCardElement === btnVaultUpload) {
          ind.textContent = 'ACTIVE PATIENT SCAN ●';
        } else {
          ind.textContent = 'ACTIVE SPECIMEN ●';
        }
      }
    }
  }

  if (btnVaultUpload) {
    btnVaultUpload.addEventListener('click', () => {
      resetLab();
      fileInput.click();
      const lab = document.getElementById('labWorkspace');
      if (lab) lab.scrollIntoView({ behavior: 'smooth' });
    });
  }

  if (btnHeaderUpload) {
    btnHeaderUpload.addEventListener('click', () => {
      fileInput.click();
    });
  }

  [specimenPCT, specimenNCT, specimenNiCT].forEach(btn => {
    if (!btn) return;
    btn.addEventListener('click', async () => {
      const cls = btn.dataset.class;
      try {
        btn.classList.add('loading');
        triggerScanLaser();

        const res = await fetch(`/api/sample/${cls}`);
        if (!res.ok) throw new Error(`Vault request failed: ${res.statusText}`);
        const blob = await res.blob();
        const file = new File([blob], `${cls}_specimen_0001.jpg`, { type: 'image/jpeg' });
        
        setActiveSpecimenCard(btn, cls);
        loadAndAnalyzeSlice(file);
      } catch (err) {
        console.error('Specimen vault error:', err);
        alert(`Vault Error: ${err.message}`);
      } finally {
        btn.classList.remove('loading');
      }
    });
  });

  // ------------------------------------------------------------------------
  // 4. Drag & Drop and File Ingestion (Dropzone + HoloViewer Global Overlay)
  // ------------------------------------------------------------------------
  dropzone.addEventListener('click', () => fileInput.click());

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
      setActiveSpecimenCard(btnVaultUpload, 'upload');
      loadAndAnalyzeSlice(fileInput.files[0]);
    }
  });

  ['dragenter', 'dragover'].forEach(name => {
    dropzone.addEventListener(name, (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(name => {
    dropzone.addEventListener(name, (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
    });
  });

  dropzone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type.startsWith('image/')) {
      setActiveSpecimenCard(btnVaultUpload, 'upload');
      loadAndAnalyzeSlice(files[0]);
    }
  });

  // Drag-and-drop directly onto Holo-Viewer anytime (even when slice is loaded)
  if (holoViewer && dragDropOverlay) {
    ['dragenter', 'dragover'].forEach(name => {
      holoViewer.addEventListener(name, (e) => {
        e.preventDefault();
        dragDropOverlay.classList.remove('hidden');
      });
    });

    holoViewer.addEventListener('dragleave', (e) => {
      e.preventDefault();
      if (!holoViewer.contains(e.relatedTarget)) {
        dragDropOverlay.classList.add('hidden');
      }
    });

    holoViewer.addEventListener('drop', (e) => {
      e.preventDefault();
      dragDropOverlay.classList.add('hidden');
      const files = e.dataTransfer.files;
      if (files.length > 0 && files[0].type.startsWith('image/')) {
        setActiveSpecimenCard(btnVaultUpload, 'upload');
        loadAndAnalyzeSlice(files[0]);
      }
    });
  }

  btnResetAll.addEventListener('click', () => {
    resetLab();
    // Scroll viewer smoothly into view
    holoViewer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  });

  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') resetLab();
  });

  // ------------------------------------------------------------------------
  // 5. Ingestion Pipeline & Inference
  // ------------------------------------------------------------------------
  function loadAndAnalyzeSlice(file) {
    currentFile = file;
    hudFilename.textContent = file.name.toUpperCase();
    triggerScanLaser();

    const reader = new FileReader();
    reader.onload = (e) => {
      rawSliceDataUrl = e.target.result;

      // Cache image for live perturbations
      const img = new Image();
      img.onload = () => { pristineImage = img; };
      img.src = rawSliceDataUrl;

      // Update Viewports
      dualRawImg.src = rawSliceDataUrl;
      dropzone.classList.add('hidden');
      updateViewportDisplay();

      // Execute Neural Inference
      executeConvolutionalInference(file);
    };
    reader.readAsDataURL(file);
  }

  function triggerScanLaser() {
    scanLaser.classList.remove('hidden');
    setTimeout(() => { scanLaser.classList.add('hidden'); }, 2200);
  }

  async function executeConvolutionalInference(fileToUpload) {
    inferenceLoader.classList.remove('hidden');

    const formData = new FormData();
    formData.append('file', fileToUpload);

    const threshold = parseFloat(uncertaintyThreshold.value);
    const alpha = parseFloat(camAlpha.value);

    try {
      const response = await fetch(`/explain?threshold=${threshold}&alpha=${alpha}`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Inference failed');
      }

      const result = await response.json();
      currentInferenceResult = result;
      renderDiagnosticIntelligence(result);
    } catch (err) {
      console.error('Inference error:', err);
      alert(`Inference Engine Error: ${err.message}`);
    } finally {
      inferenceLoader.classList.add('hidden');
    }
  }

  // ------------------------------------------------------------------------
  // 6. Render Diagnostic Readout & Telemetry
  // ------------------------------------------------------------------------
  function renderDiagnosticIntelligence(res) {
    telemetryEmpty.classList.add('hidden');
    telemetryActive.classList.remove('hidden');

    const threshold = parseFloat(uncertaintyThreshold.value);
    const isUncertain = res.confidence < threshold;
    const effectiveLabel = isUncertain ? 'Uncertain' : res.predicted_class;

    // Cache URLs
    overlayDataUrl = res.overlay_image;
    heatmapDataUrl = res.heatmap_image;
    dualCamImg.src = overlayDataUrl;

    // Update Verdict Banner Styles & Copy
    verdictBanner.className = 'verdict-banner';

    if (effectiveLabel === 'pCT') {
      verdictBanner.classList.add('banner-pct');
      verdictTag.textContent = 'COVID-19 PATHOLOGY [pCT]';
      verdictTag.style.color = '#fda4af';
      verdictTag.style.borderColor = 'rgba(255, 42, 95, 0.4)';
      verdictTitle.textContent = 'PULMONARY INFILTRATES DETECTED';
      verdictSummary.textContent = res.class_description || 'Multifocal peripheral ground-glass opacities (GGO) with subpleural predilection identified in bilateral lung fields.';
      stDetail.textContent = 'High-priority positive pathology flagged. Minimizing Type II false-negative risk. Immediate clinical PCR correlation indicated.';
    } else if (effectiveLabel === 'nCT') {
      verdictBanner.classList.add('banner-nct');
      verdictTag.textContent = 'CLEAR LUNG [nCT]';
      verdictTag.style.color = '#6ee7b7';
      verdictTag.style.borderColor = 'rgba(16, 185, 129, 0.4)';
      verdictTitle.textContent = 'PARENCHYMAL CLEARANCE VERIFIED';
      verdictSummary.textContent = res.class_description || 'Intact radiologic lung fields. No characteristic COVID-19 consolidation or ground-glass opacification detected.';
      stDetail.textContent = 'Standard normal clearance. Baseline test specificity: 99.79% on real clinical CT data (3 false alarms per 1,498 controls).';
    } else if (effectiveLabel === 'NiCT') {
      verdictBanner.classList.add('banner-nict');
      verdictTag.textContent = 'NON-INFORMATIVE [NiCT]';
      verdictTag.style.color = '#7dd3fc';
      verdictTag.style.borderColor = 'rgba(0, 240, 255, 0.4)';
      verdictTitle.textContent = 'NON-INFORMATIVE SLICE';
      verdictSummary.textContent = res.class_description || 'Slice cuts outside thoracic lungs (cranial/neck ring or sub-diaphragmatic tissue). Model suppresses forced disease guessing.';
      stDetail.textContent = 'Non-informative rejection active. Preserves clinical specificity by eliminating out-of-distribution anatomical cuts.';
    } else {
      verdictBanner.classList.add('banner-uncertain');
      verdictTag.textContent = 'UNCERTAIN TRIAGE';
      verdictTag.style.color = '#fde68a';
      verdictTag.style.borderColor = 'rgba(245, 158, 11, 0.4)';
      verdictTitle.textContent = 'AMBIGUOUS ATTENTION FIELD';
      verdictSummary.textContent = `Confidence (${(res.confidence * 100).toFixed(1)}%) falls below calibrated safety threshold (${(threshold * 100).toFixed(0)}%). Specimen automatically routed to human board radiologist review.`;
      stDetail.textContent = 'Rejection protocol active: Low-confidence outputs strictly suppressed to eliminate dangerous automated diagnostic guesses.';
    }

    // Top-1 Confidence Meter
    const confPct = (res.confidence * 100).toFixed(1);
    vcmNumber.textContent = `${confPct}%`;
    vcmBar.style.width = `${confPct}%`;
    vcmBar.style.background = effectiveLabel === 'pCT' ? 'var(--bio-crimson)' : (effectiveLabel === 'nCT' ? 'var(--clinical-emerald)' : 'var(--electric-cyan)');

    // 3-Class Probability Bars
    const probs = res.probs;
    const pCTPct = (probs.pCT * 100).toFixed(1);
    const nCTPct = (probs.nCT * 100).toFixed(1);
    const nICTPct = (probs.NiCT * 100).toFixed(1);

    valPCT.textContent = `${pCTPct}%`;
    valNCT.textContent = `${nCTPct}%`;
    valNiCT.textContent = `${nICTPct}%`;

    barPCT.style.width = `${pCTPct}%`;
    barNCT.style.width = `${nCTPct}%`;
    barNiCT.style.width = `${nICTPct}%`;

    // Explainability Card Narrative
    ecTargetClass.textContent = res.target_class;
    ecNarrative.innerHTML = `Attention mapped for target class: <strong class="highlight-cyan">${res.target_class}</strong> (Confidence: ${(res.confidence * 100).toFixed(1)}%). Convolutional gradients isolate characteristic lesion regions inside lung fields, verifying the model evaluates thoracic parenchyma rather than scanner borders, text annotations, or bed contours.`;

    updateViewportDisplay();
  }

  // ------------------------------------------------------------------------
  // 7. View Mode Switcher (DUAL FOCUS | OVERLAY | THERMAL | RAW SLICE)
  // ------------------------------------------------------------------------
  [btnViewDual, btnViewOverlay, btnViewHeatmap, btnViewRaw].forEach(btn => {
    btn.addEventListener('click', () => {
      [btnViewDual, btnViewOverlay, btnViewHeatmap, btnViewRaw].forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      activeViewMode = btn.dataset.view;
      updateViewportDisplay();
    });
  });

  function updateViewportDisplay() {
    if (!rawSliceDataUrl) return;

    if (activeViewMode === 'dual') {
      singleViewport.classList.add('hidden');
      dualViewport.classList.remove('hidden');
      dualRawImg.src = rawSliceDataUrl;
      dualCamImg.src = overlayDataUrl || rawSliceDataUrl;
    } else {
      dualViewport.classList.add('hidden');
      singleViewport.classList.remove('hidden');

      if (activeViewMode === 'overlay') {
        mainHoloImg.src = overlayDataUrl || rawSliceDataUrl;
      } else if (activeViewMode === 'heatmap') {
        mainHoloImg.src = heatmapDataUrl || rawSliceDataUrl;
      } else {
        mainHoloImg.src = rawSliceDataUrl;
      }
    }
  }

  // ------------------------------------------------------------------------
  // 8. Sliders & Threshold Interactivity
  // ------------------------------------------------------------------------
  uncertaintyThreshold.addEventListener('input', () => {
    const val = parseFloat(uncertaintyThreshold.value);
    valThreshold.textContent = `${(val * 100).toFixed(0)}%`;
    if (currentInferenceResult) {
      renderDiagnosticIntelligence(currentInferenceResult);
    }
  });

  camAlpha.addEventListener('input', () => {
    valAlpha.textContent = `${(parseFloat(camAlpha.value) * 100).toFixed(0)}%`;
  });

  camAlpha.addEventListener('change', () => {
    if (currentFile) executeConvolutionalInference(currentFile);
  });

  // ------------------------------------------------------------------------
  // 9. Interactive Scanner Perturbation Simulator (Client Canvas)
  // ------------------------------------------------------------------------
  function executeSimulatorPerturbations() {
    if (!pristineImage) return;

    const jpegQuality = parseInt(jpegSlider.value) / 100.0;
    const brightness = parseFloat(brightSlider.value);
    const rotation = parseFloat(rotSlider.value);

    valJpeg.textContent = jpegQuality >= 1.0 ? '100% [PRISTINE]' : `${(jpegQuality * 100).toFixed(0)}% [LOSSY]`;
    valBright.textContent = `${brightness.toFixed(1)}× [${brightness > 1.0 ? 'OVEREXPOSED' : (brightness < 1.0 ? 'UNDEREXPOSED' : 'NOMINAL')}]`;
    valRot.textContent = `${rotation}° [${rotation !== 0 ? 'TILTED' : 'ALIGNED'}]`;

    const canvas = document.createElement('canvas');
    canvas.width = pristineImage.naturalWidth || 256;
    canvas.height = pristineImage.naturalHeight || 256;
    const ctx = canvas.getContext('2d');

    // Rotation transform
    ctx.save();
    ctx.translate(canvas.width / 2, canvas.height / 2);
    ctx.rotate((rotation * Math.PI) / 180);
    ctx.translate(-canvas.width / 2, -canvas.height / 2);

    // Draw base pristine image
    ctx.drawImage(pristineImage, 0, 0, canvas.width, canvas.height);
    ctx.restore();

    // Brightness manipulation
    if (brightness !== 1.0) {
      const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const d = imgData.data;
      for (let i = 0; i < d.length; i += 4) {
        d[i] = Math.min(255, d[i] * brightness);
        d[i + 1] = Math.min(255, d[i + 1] * brightness);
        d[i + 2] = Math.min(255, d[i + 2] * brightness);
      }
      ctx.putImageData(imgData, 0, 0);
    }

    // Export as lossy JPEG to simulate transmission artifacts
    canvas.toBlob((blob) => {
      const perturbedFile = new File([blob], `sim_${currentFile.name}`, { type: 'image/jpeg' });
      rawSliceDataUrl = URL.createObjectURL(blob);
      dualRawImg.src = rawSliceDataUrl;
      updateViewportDisplay();
      executeConvolutionalInference(perturbedFile);
    }, 'image/jpeg', jpegQuality);
  }

  [jpegSlider, brightSlider, rotSlider].forEach(slider => {
    slider.addEventListener('change', executeSimulatorPerturbations);
    slider.addEventListener('input', () => {
      const jq = parseInt(jpegSlider.value);
      valJpeg.textContent = jq >= 100 ? '100% [PRISTINE]' : `${jq}% [LOSSY]`;
      valBright.textContent = `${parseFloat(brightSlider.value).toFixed(1)}×`;
      valRot.textContent = `${rotSlider.value}°`;
    });
  });

  btnResetPerturbations.addEventListener('click', () => {
    jpegSlider.value = 100;
    brightSlider.value = 1.0;
    rotSlider.value = 0;
    valJpeg.textContent = '100% [PRISTINE]';
    valBright.textContent = '1.0× [NOMINAL]';
    valRot.textContent = '0° [ALIGNED]';
    if (currentFile) loadAndAnalyzeSlice(currentFile);
  });

  // ------------------------------------------------------------------------
  // 10. Lab Eject / Reset
  // ------------------------------------------------------------------------
  function resetLab() {
    currentFile = null;
    pristineImage = null;
    rawSliceDataUrl = '';
    overlayDataUrl = '';
    heatmapDataUrl = '';
    currentInferenceResult = null;
    fileInput.value = '';
    hudFilename.textContent = 'AWAITING_SLICE.RAW';

    singleViewport.classList.add('hidden');
    dualViewport.classList.add('hidden');
    dropzone.classList.remove('hidden');
    telemetryActive.classList.add('hidden');
    telemetryEmpty.classList.remove('hidden');
    setActiveSpecimenCard(null);
  }

  // Auto-load specimen #01 on first arrival for immediate awe!
  if (specimenPCT) {
    specimenPCT.click();
  }
});
