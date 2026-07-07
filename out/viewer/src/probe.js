// Diagnostic probe. Query params:
//   a    = front|profile|back|threeq   camera angle
//   mode = hard   : keep GLB materials, force opaque + DoubleSide (what the viewer does)
//          colors : replace each material with a distinct SOLID OPAQUE color
//                   (HeadMat=red, EyeMat=green, RestMat=blue) -> see-through is unmistakable
//          flat   : one opaque grey for all
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const q = new URLSearchParams(location.search);
const angle = q.get('a') || 'profile';
const mode = q.get('mode') || 'hard';
const W = 1100, H = 820;

function log(msg) {
  const p = document.createElement('pre');
  p.textContent = msg;
  p.style.cssText = 'position:fixed;left:4px;top:4px;color:#7f7;font:11px monospace;margin:0;white-space:pre-wrap';
  document.body.appendChild(p);
  document.title = 'PROBE ' + msg.slice(0, 120);
}

async function run() {
  const scene = new THREE.Scene();
  const bg = q.get('bg');
  scene.background = new THREE.Color(bg === 'magenta' ? 0xff00ff : bg === 'green' ? 0x00ff00 : 0x14161c);
  const cam = new THREE.PerspectiveCamera(35, W / H, 0.001, 100);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(W, H);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  document.body.appendChild(renderer.domElement);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x404040, 1.3));
  const key = new THREE.DirectionalLight(0xffffff, 2.0); key.position.set(1, 1.2, 2); scene.add(key);

  const loader = new GLTFLoader();
  const gltf = await loader.loadAsync('/head_arkit_v2.glb?cb=' + Date.now());
  scene.add(gltf.scene);

  const colorFor = { HeadMat: 0xd23b2f, EyeMat: 0x2fd23b, RestMat: 0x2f6fd2 };
  const report = [];
  gltf.scene.traverse((o) => {
    if (!o.isMesh) return;
    const ms = Array.isArray(o.material) ? o.material : [o.material];
    ms.forEach((m) => {
      if (!m) return;
      report.push(`${o.name}/${m.name} in:{transp:${m.transparent},side:${m.side},blend:${m.blending},op:${m.opacity},vcol:${m.vertexColors}}`);
    });
    if (mode === 'colors') {
      o.material = new THREE.MeshStandardMaterial({
        color: colorFor[ms[0]?.name] ?? 0x888888, roughness: 0.8, side: THREE.DoubleSide,
      });
    } else if (mode === 'flat') {
      o.material = new THREE.MeshStandardMaterial({ color: 0xb8b0a8, roughness: 0.7, side: THREE.DoubleSide });
    } else {
      ms.forEach((m) => {
        if (!m) return;
        m.transparent = false; m.opacity = 1; m.alphaTest = 0;
        m.depthWrite = true; m.depthTest = true; m.side = THREE.DoubleSide;
        m.needsUpdate = true;
      });
    }
  });

  const box = new THREE.Box3().setFromObject(gltf.scene);
  const c = box.getCenter(new THREE.Vector3());
  const s = box.getSize(new THREE.Vector3());
  gltf.scene.position.sub(c);
  const d = Math.max(s.x, s.y, s.z) || 0.2;
  const cams = {
    front: [0, 0, d * 1.8], profile: [d * 1.9, 0, d * 0.15],
    back: [0, 0, -d * 1.9], threeq: [d * 1.4, d * 0.15, d * 1.2],
  };
  cam.position.set(...(cams[angle] || cams.profile));
  cam.lookAt(0, 0, 0);
  renderer.render(scene, cam);
  log(`mode=${mode} a=${angle}\n` + report.join('\n'));
}

run().catch((e) => log('ERROR ' + (e && e.stack || e)));
