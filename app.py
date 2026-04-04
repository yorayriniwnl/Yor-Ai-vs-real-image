import os
import base64
import binascii
import json

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import joblib
import cv2
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from sklearn.model_selection import train_test_split

MAX_UPLOAD_MB = 25
UPLOAD_HINT = f"Limit {MAX_UPLOAD_MB}MB per file - JPG, PNG, JPEG"

# -------------------------
# Load Model & Scaler
# -------------------------
@st.cache_resource
def load_model():
    model = joblib.load("svm_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_model()
except Exception:
    model, scaler = None, None

# -----------------------------
# Feature Extraction Function
# -----------------------------
def extract_features_from_gray(gray):
    if gray is None:
        return None

    gray = cv2.resize(gray, (224, 224))
    norm_image = gray / 255.0

    variance = np.var(norm_image)
    laplacian = cv2.Laplacian(norm_image, cv2.CV_64F)
    hf_variance = np.var(laplacian)

    radius = 1
    n_points = 8 * radius

    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, n_points + 3),
        range=(0, n_points + 2)
    )
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)

    reduced = gray // 16
    glcm = graycomatrix(
        reduced,
        distances=[1],
        angles=[0],
        levels=16,
        symmetric=True,
        normed=True
    )

    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    glcm_matrix = glcm[:, :, 0, 0]
    entropy = -np.sum(glcm_matrix * np.log2(glcm_matrix + 1e-10))

    feature_vector = np.hstack([
        variance,
        hf_variance,
        hist,
        contrast,
        energy,
        homogeneity,
        correlation,
        entropy
    ])

    return feature_vector


def extract_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return extract_features_from_gray(image)


def extract_features_from_image(image_bgr):
    if image_bgr is None:
        return None
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return extract_features_from_gray(gray)


def predict_probabilities(model_obj, features_scaled):
    if model_obj is None:
        return 0.5, 0.5

    if hasattr(model_obj, "predict_proba"):
        proba = model_obj.predict_proba(features_scaled)[0]
        classes = getattr(model_obj, "classes_", None)
        if classes is not None and 1 in classes:
            ai_idx = list(classes).index(1)
        elif len(proba) > 1:
            ai_idx = 1
        else:
            ai_idx = 0
        prob_ai = float(proba[ai_idx])
        confidence = float(max(prob_ai, 1.0 - prob_ai))
        return prob_ai, confidence

    if hasattr(model_obj, "decision_function"):
        score = model_obj.decision_function(features_scaled)
        score = float(score[0]) if np.ndim(score) else float(score)
        score = float(np.clip(score, -10.0, 10.0))
        prob_ai = float(1.0 / (1.0 + np.exp(-score)))
        confidence = float(max(prob_ai, 1.0 - prob_ai))
        return prob_ai, confidence

    pred = model_obj.predict(features_scaled)[0]
    prob_ai = 1.0 if pred == 1 else 0.0
    confidence = 0.5
    return float(prob_ai), float(confidence)


def image_bytes_to_data_url(file_bytes, mime_type):
    if not mime_type:
        mime_type = "image/jpeg"
    encoded = base64.b64encode(file_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


# -----------------------------
# Build Dataset
# -----------------------------
@st.cache_data
def build_dataset_counts():
    X = []
    y = []

    dataset_path = "dataset"

    for label_folder in ["real", "ai"]:
        folder_path = os.path.join(dataset_path, label_folder)

        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)

            features = extract_features(file_path)

            if features is not None:
                X.append(features)

                if label_folder == "real":
                    y.append(0)
                else:
                    y.append(1)

    X = np.array(X)
    y = np.array(y)

    print("Total samples:", len(X))

    # -----------------------------
    # PROCESS 10: Train-Test Split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    return len(X), len(X_train), len(X_test)


UI_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>AI vs Real Image Detector</title>
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  body{background:#05070F;overflow:hidden;font-family:'SF Pro Display','Segoe UI',system-ui,sans-serif}
  #root{position:fixed;inset:0;width:100%;height:100%}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.2}}
  @keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-10px)}}
  @keyframes scanLine{0%{top:-4px}100%{top:calc(100% + 4px)}}
  @keyframes glitch{0%{transform:translate(0)}20%{transform:translate(-3px,2px)}40%{transform:translate(3px,-2px)}60%{transform:translate(-2px,0)}80%{transform:translate(0,2px)}100%{transform:translate(0)}}
  @keyframes ripple{0%{transform:scale(0);opacity:.8}100%{transform:scale(4);opacity:0}}
  @keyframes shimmer{0%{left:-60%}100%{left:160%}}
  @keyframes spin{to{transform:rotate(360deg)}}
  @keyframes breathe{0%,100%{box-shadow:0 0 20px rgba(0,229,168,.05),0 0 60px rgba(91,140,255,.03)}50%{box-shadow:0 0 40px rgba(0,229,168,.12),0 0 80px rgba(91,140,255,.06)}}
  @keyframes trailFade{0%{opacity:.8;transform:scale(1)}100%{opacity:0;transform:scale(.5)}}
  @keyframes dataScroll{0%{transform:translateY(0)}100%{transform:translateY(-50%)}}
  canvas{display:block}
</style>
</head>
<body>
<div id="root"></div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/react/18.2.0/umd/react.production.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/react-dom/18.2.0/umd/react-dom.production.min.js"></script>

<script>
const INITIAL_DATA = __INITIAL_DATA__;
const {useState,useEffect,useRef,useCallback,useMemo}=React;
const h=React.createElement;

const C={primary:'#00E5A8',secondary:'#5B8CFF',danger:'#FF4D4D',bg:'#05070F',bgDark:'#020409'};

// ── THREE.JS BACKGROUND ──────────────────────────────────────────────────────
function initThree(canvas, mouseRef){
  if(!canvas)return()=>{};
  const lowEnd=(navigator.deviceMemory&&navigator.deviceMemory<=4)||(navigator.hardwareConcurrency&&navigator.hardwareConcurrency<=4);
  const W=window.innerWidth,H=window.innerHeight;
  const renderer=new THREE.WebGLRenderer({canvas,antialias:true,alpha:false,powerPreference:'high-performance'});
  const pixelRatio=Math.min(devicePixelRatio,lowEnd?1.25:2);
  renderer.setPixelRatio(pixelRatio);
  renderer.setSize(W,H);
  renderer.setClearColor(0x05070F,1);

  const scene=new THREE.Scene();
  scene.fog=new THREE.FogExp2(0x020409,.008);
  const camera=new THREE.PerspectiveCamera(60,W/H,.1,1000);
  camera.position.z=80;

  // Particles
  const N=lowEnd?3500:7000;
  const pos=new Float32Array(N*3),col=new Float32Array(N*3);
  for(let i=0;i<N;i++){
    const r=Math.random()*220,theta=Math.random()*Math.PI*2,phi=Math.acos(2*Math.random()-1);
    pos[i*3]=r*Math.sin(phi)*Math.cos(theta);
    pos[i*3+1]=r*Math.sin(phi)*Math.sin(theta);
    pos[i*3+2]=r*Math.cos(phi);
    const t=Math.random();
    if(t<.4){col[i*3]=0;col[i*3+1]=.9;col[i*3+2]=.66;}
    else if(t<.7){col[i*3]=.36;col[i*3+1]=.55;col[i*3+2]=1;}
    else{col[i*3]=.2;col[i*3+1]=.3;col[i*3+2]=.5;}
  }
  const pGeo=new THREE.BufferGeometry();
  pGeo.setAttribute('position',new THREE.BufferAttribute(pos,3));
  pGeo.setAttribute('color',new THREE.BufferAttribute(col,3));
  const pMat=new THREE.PointsMaterial({size:.35,vertexColors:true,transparent:true,opacity:.75,sizeAttenuation:true});
  const particles=new THREE.Points(pGeo,pMat);
  scene.add(particles);

  // Neural network nodes
  const nodes=[];
  const nodeCount=lowEnd?45:70;
  for(let i=0;i<nodeCount;i++){
    nodes.push(new THREE.Vector3((Math.random()-.5)*200,(Math.random()-.5)*120,(Math.random()-.5)*80-10));
  }

  // Neural lines
  const lineGroup=new THREE.Group();
  const lineData=[];
  const lineDist=lowEnd?38:45;
  const lineChance=lowEnd?0.08:0.12;
  nodes.forEach((a,ai)=>{
    nodes.forEach((b,bi)=>{
      if(bi<=ai)return;
      if(a.distanceTo(b)<lineDist&&Math.random()<lineChance){
        const g=new THREE.BufferGeometry().setFromPoints([a,b]);
        const m=new THREE.LineBasicMaterial({color:Math.random()>.5?0x00E5A8:0x5B8CFF,transparent:true,opacity:.04});
        lineGroup.add(new THREE.Line(g,m));
        lineData.push({base:.02+Math.random()*.06,phase:Math.random()*Math.PI*2,spd:.5+Math.random()*2});
      }
    });
  });
  scene.add(lineGroup);

  // Node spheres
  const nGroup=new THREE.Group();
  nodes.forEach(v=>{
    const s=new THREE.Mesh(
      new THREE.SphereGeometry(.4,6,6),
      new THREE.MeshBasicMaterial({color:Math.random()>.5?0x00E5A8:0x5B8CFF,transparent:true,opacity:.3})
    );
    s.position.copy(v);nGroup.add(s);
  });
  scene.add(nGroup);

  // Floating wireframe orbs
  const orbGroup=new THREE.Group();
  [{c:0x00E5A8,x:40,y:-20,z:-30,r:8},{c:0x5B8CFF,x:-50,y:30,z:-50,r:12},{c:0x00E5A8,x:0,y:0,z:-60,r:15}].forEach(o=>{
    const mesh=new THREE.Mesh(
      new THREE.SphereGeometry(o.r,24,24),
      new THREE.MeshBasicMaterial({color:o.c,transparent:true,opacity:.025,wireframe:true})
    );
    mesh.position.set(o.x,o.y,o.z);orbGroup.add(mesh);
  });
  scene.add(orbGroup);

  const pl1=new THREE.PointLight(0x00E5A8,2,150);pl1.position.set(50,30,20);scene.add(pl1);
  const pl2=new THREE.PointLight(0x5B8CFF,1.5,150);pl2.position.set(-50,-20,10);scene.add(pl2);

  let frame=0,id;
  const tick=()=>{
    id=requestAnimationFrame(tick);frame++;
    const t=frame*.001;
    particles.rotation.y=t*.03;particles.rotation.x=t*.012;
    const mx=mouseRef.current.x,my=mouseRef.current.y;
    camera.position.x=Math.sin(t*.18)*10+mx*15;
    camera.position.y=Math.cos(t*.14)*6-my*10;
    camera.lookAt(0,0,0);
    lineGroup.children.forEach((l,i)=>{const d=lineData[i];if(d)l.material.opacity=d.base*(.5+.5*Math.sin(t*d.spd*2+d.phase));});
    nGroup.children.forEach((n,i)=>n.material.opacity=.05+.25*(.5+.5*Math.sin(t*1.2+i*.4)));
    orbGroup.children.forEach((o,i)=>{o.rotation.x=t*.15+i;o.rotation.y=t*.1+i*.5;});
    pl1.intensity=1.5+Math.sin(t*1.7)*.5;pl2.intensity=1+Math.cos(t*1.3)*.4;
    renderer.render(scene,camera);
  };
  tick();

  const onResize=()=>{
    const W=window.innerWidth,H=window.innerHeight;
    camera.aspect=W/H;camera.updateProjectionMatrix();renderer.setSize(W,H);
  };
  window.addEventListener('resize',onResize);
  const disposeMesh=mesh=>{
    if(!mesh)return;
    if(mesh.geometry)mesh.geometry.dispose();
    if(mesh.material){
      if(Array.isArray(mesh.material)){mesh.material.forEach(m=>m.dispose());}
      else{mesh.material.dispose();}
    }
  };
  return()=>{
    cancelAnimationFrame(id);
    window.removeEventListener('resize',onResize);
    lineGroup.children.forEach(disposeMesh);
    nGroup.children.forEach(disposeMesh);
    orbGroup.children.forEach(disposeMesh);
    pGeo.dispose();
    pMat.dispose();
    renderer.dispose();
  };
}

// ── RADIAL CONFIDENCE RING ───────────────────────────────────────────────────
function RadialRing({value,color,size=130}){
  const r=(size-14)/2,circ=r*2*Math.PI;
  const [anim,setAnim]=useState(circ);
  useEffect(()=>{
    const target=circ-(value/100)*circ;
    let start=null,from=circ;
    const go=ts=>{
      if(!start){start=ts;}
      const p=Math.min((ts-start)/1800,1);
      const ease=1-Math.pow(1-p,3);
      setAnim(from+(target-from)*ease);
      if(p<1)requestAnimationFrame(go);
    };
    const delay=setTimeout(()=>requestAnimationFrame(go),300);
    return()=>clearTimeout(delay);
  },[value]);
  return h('div',{style:{width:size,height:size,position:'relative',flexShrink:0}},
    h('svg',{width:size,height:size,style:{transform:'rotate(-90deg)',overflow:'visible'}},
      h('circle',{cx:size/2,cy:size/2,r,fill:'none',stroke:'rgba(255,255,255,0.04)',strokeWidth:6}),
      h('circle',{cx:size/2,cy:size/2,r,fill:'none',stroke:color+'22',strokeWidth:10}),
      h('circle',{cx:size/2,cy:size/2,r,fill:'none',stroke:color,strokeWidth:3.5,strokeLinecap:'round',
        strokeDasharray:circ,strokeDashoffset:anim,style:{filter:`drop-shadow(0 0 8px ${color})`}})
    ),
    h('div',{style:{position:'absolute',inset:0,display:'flex',flexDirection:'column',alignItems:'center',justifyContent:'center',gap:2}},
      h('div',{style:{color,fontSize:'1.9rem',fontWeight:700,fontFamily:'monospace',lineHeight:1}},[value,'%']),
      h('div',{style:{color:'rgba(255,255,255,.3)',fontSize:'0.55rem',letterSpacing:'.15em',fontFamily:'monospace'}},'CONFIDENCE')
    )
  );
}

// ── AI SCAN CANVAS ANIMATION ─────────────────────────────────────────────────
function ScanCanvas({imageUrl,onComplete}){
  const ref=useRef(null);
  useEffect(()=>{
    const canvas=ref.current;if(!canvas)return;
    const ctx=canvas.getContext('2d');
    const img=new Image();
    img.onload=()=>{
      const W=canvas.width,H=canvas.height;
      let frame=0;const total=150;let id;
      const corners=[
        {x:15,y:15,w:40,h:40},{x:W-55,y:15,w:40,h:40},
        {x:15,y:H-55,w:40,h:40},{x:W-55,y:H-55,w:40,h:40},
        {x:W/2-30,y:H/2-30,w:60,h:60}
      ];
      const tick=()=>{
        frame++;const scanY=(frame/total)*H;
        ctx.clearRect(0,0,W,H);
        ctx.drawImage(img,0,0,W,H);
        // dark unscanned region
        ctx.fillStyle='rgba(2,4,9,.5)';ctx.fillRect(0,scanY,W,H-scanY);
        // grid overlay
        ctx.strokeStyle='rgba(0,229,168,.12)';ctx.lineWidth=.5;
        const g=20;
        for(let x=0;x<W;x+=g){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,Math.min(scanY,H));ctx.stroke();}
        for(let y=0;y<=Math.min(scanY,H);y+=g){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke();}
        // corner brackets
        if(frame>20){
          const a=Math.min((frame-20)/25,.85);ctx.strokeStyle=`rgba(0,229,168,${a})`;ctx.lineWidth=1.5;
          corners.forEach(c=>{
            const cs=12;ctx.beginPath();
            ctx.moveTo(c.x+cs,c.y);ctx.lineTo(c.x,c.y);ctx.lineTo(c.x,c.y+cs);
            ctx.moveTo(c.x+c.w-cs,c.y);ctx.lineTo(c.x+c.w,c.y);ctx.lineTo(c.x+c.w,c.y+cs);
            ctx.moveTo(c.x,c.y+c.h-cs);ctx.lineTo(c.x,c.y+c.h);ctx.lineTo(c.x+cs,c.y+c.h);
            ctx.moveTo(c.x+c.w-cs,c.y+c.h);ctx.lineTo(c.x+c.w,c.y+c.h);ctx.lineTo(c.x+c.w,c.y+c.h-cs);
            ctx.stroke();
          });
        }
        // scan beam gradient
        if(scanY<H){
          const gr=ctx.createLinearGradient(0,scanY-35,0,scanY+5);
          gr.addColorStop(0,'rgba(0,229,168,0)');
          gr.addColorStop(.7,'rgba(0,229,168,.2)');
          gr.addColorStop(1,'rgba(0,229,168,.6)');
          ctx.fillStyle=gr;ctx.fillRect(0,scanY-35,W,40);
          ctx.strokeStyle='rgba(0,229,168,.95)';ctx.lineWidth=1.5;
          ctx.beginPath();ctx.moveTo(0,scanY);ctx.lineTo(W,scanY);ctx.stroke();
          // scan particles
          for(let px=0;px<W;px+=12){
            if(Math.random()>.65){
              ctx.fillStyle=`rgba(0,229,168,${Math.random()*.7+.2})`;
              ctx.beginPath();ctx.arc(px+Math.random()*8,scanY+(Math.random()-.5)*4,Math.random()*1.5+.5,0,Math.PI*2);ctx.fill();
            }
          }
        }
        // data stream text
        if(frame>50){
          ctx.font='9px monospace';
          for(let di=0;di<4;di++){
            ctx.fillStyle=`rgba(0,229,168,${Math.random()*.35})`;
            ctx.fillText(Math.random().toString(36).slice(2,8).toUpperCase(),di%2===0?4:W-44,Math.random()*scanY);
          }
        }
        // status text
        ctx.font='bold 10px monospace';ctx.fillStyle='rgba(0,229,168,.9)';
        ctx.fillText(`ANALYZING... ${Math.floor(frame/total*100)}%`,10,H-10);
        ctx.fillStyle='rgba(91,140,255,.7)';
        ctx.fillText(`FREQ SCAN`,W-72,H-10);
        if(frame<total)id=requestAnimationFrame(tick);else onComplete();
      };
      id=requestAnimationFrame(tick);
    };
    img.src=imageUrl;
  },[imageUrl]);
  return h('canvas',{ref,width:400,height:260,style:{width:'100%',height:'100%',borderRadius:8}});
}

// ── FLOATING PARTICLES ───────────────────────────────────────────────────────
function FloatingParticles({active}){
  const particles=useMemo(()=>Array.from({length:12},(_,i)=>({
    id:i,x:Math.random()*100,y:Math.random()*100,
    size:Math.random()*3+1,dur:Math.random()*4+3,delay:Math.random()*3,
    color:Math.random()>.5?C.primary:C.secondary
  })),[]);
  if(!active)return null;
  return h('div',{style:{position:'absolute',inset:0,pointerEvents:'none',overflow:'hidden',borderRadius:'inherit'}},
    particles.map(p=>h('div',{key:p.id,style:{
      position:'absolute',left:p.x+'%',top:p.y+'%',
      width:p.size,height:p.size,borderRadius:'50%',background:p.color,
      opacity:.6,animation:`trailFade ${p.dur}s ${p.delay}s infinite ease-out`,
      boxShadow:`0 0 ${p.size*3}px ${p.color}`
    }}))
  );
}

// ── MAIN APP ─────────────────────────────────────────────────────────────────
function App(){
  const initial=INITIAL_DATA||{};
  const bgRef=useRef(null);
  const mouseRef=useRef({x:0,y:0});
  const cardRef=useRef(null);
  const [stage,setStage]=useState(initial.stage||'idle');
  const [result,setResult]=useState(initial.result||null);
  const [imageUrl,setImageUrl]=useState(initial.imageUrl||null);
  const [dragging,setDragging]=useState(false);
  const [burst,setBurst]=useState(false);
  const [glitch,setGlitch]=useState(false);
  const [cardRot,setCardRot]=useState({x:0,y:0});
  const rotAnim=useRef({x:0,y:0,tx:0,ty:0,id:null});
  const hideResetButton=!!initial.hideResetButton;

  // Init Three.js
  useEffect(()=>{
    if(bgRef.current){
      const cleanup=initThree(bgRef.current,mouseRef);
      return cleanup;
    }
  },[]);

  // Mouse parallax
  useEffect(()=>{
    const mm=e=>{
      mouseRef.current={x:(e.clientX/window.innerWidth)*2-1,y:(e.clientY/window.innerHeight)*2-1};
    };
    window.addEventListener('mousemove',mm);
    return()=>window.removeEventListener('mousemove',mm);
  },[]);

  // Card tilt spring
  useEffect(()=>{
    const loop=()=>{
      const r=rotAnim.current;
      r.x+=(r.tx-r.x)*.08;r.y+=(r.ty-r.y)*.08;
      setCardRot({x:r.x,y:r.y});
      r.id=requestAnimationFrame(loop);
    };
    rotAnim.current.id=requestAnimationFrame(loop);
    return()=>cancelAnimationFrame(rotAnim.current.id);
  },[]);

  const onCardMove=useCallback(e=>{
    if(!cardRef.current)return;
    const r=cardRef.current.getBoundingClientRect();
    rotAnim.current.tx=((e.clientX-(r.left+r.width/2))/(r.width/2))*9;
    rotAnim.current.ty=-((e.clientY-(r.top+r.height/2))/(r.height/2))*7;
  },[]);

  const onCardLeave=useCallback(()=>{rotAnim.current.tx=0;rotAnim.current.ty=0;},[]);

  const processFile=useCallback(file=>{
    return;
  },[]);

  const onScanDone=useCallback(()=>{
    if(initial && initial.result && initial.imageUrl){
      setResult(initial.result);setStage('result');
      if(!initial.result.isReal){
        setGlitch(true);let c=0;
        const iv=setInterval(()=>{setGlitch(v=>!v);if(++c>12)clearInterval(iv);},120);
      }
    }
  },[]);

  const reset=useCallback(()=>{
    setStage('idle');setResult(null);setGlitch(false);
    if(imageUrl&&imageUrl.startsWith('blob:'))URL.revokeObjectURL(imageUrl);
    setImageUrl(null);
  },[imageUrl]);

  const rc=result?(result.isReal?C.primary:C.danger):C.primary;

  const stats=initial.stats||[];
  const showStats=stats.length>0;
  const showTrend=!!initial.showTrend;
  const recentScans=initial.recentScans||[];
  const showRecent=recentScans.length>0;
  const showSystem=!!initial.showSystem;
  const systemStatus=initial.systemStatus||[];
  const metrics=initial.metrics||[];
  const showMetrics=metrics.length>0;
  const heatmapUrl=initial.heatmapUrl||null;
  const resultSummary=initial.resultSummary||'';
  const modelName=initial.modelName||'';
  const uploadHint=initial.uploadHint||'';
  const badges=initial.badges||[];
  const showBadges=badges.length>0;
  const footerItems=initial.footerItems||[];
  const footerNote=initial.footerNote||'';
  const showFooter=footerItems.length>0||!!footerNote;
  const summaryText=resultSummary||(result?(result.isReal?'No strong generative artifacts detected':'High likelihood of synthetic artifacts'):'');

  return h('div',{style:{position:'fixed',inset:0,background:C.bg,overflow:'hidden',fontFamily:"'SF Pro Display','Segoe UI',system-ui,sans-serif"}},
    // WebGL canvas
    h('canvas',{ref:bgRef,style:{position:'absolute',inset:0,zIndex:0,width:'100%',height:'100%'}}),
    // Vignette overlay
    h('div',{style:{position:'absolute',inset:0,zIndex:1,pointerEvents:'none',background:'radial-gradient(ellipse 80% 80% at 50% 50%, transparent 20%, rgba(2,4,9,.9) 100%)'}}),
    // Scanlines
    h('div',{style:{position:'absolute',inset:0,zIndex:1,pointerEvents:'none',backgroundImage:'repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,.04) 2px,rgba(0,0,0,.04) 4px)'}}),

    // UI layer
    h('div',{style:{position:'relative',zIndex:10,width:'100%',height:'100%',display:'flex',flexDirection:'column'}},

      // ── HEADER ──
      h('header',{style:{padding:'16px 28px',display:'flex',alignItems:'center',justifyContent:'space-between',borderBottom:'1px solid rgba(0,229,168,.07)',backdropFilter:'blur(20px)',animation:'fadeDown .8s ease both'}},
        h('div',{style:{display:'flex',alignItems:'center',gap:12}},
          h('div',{style:{width:38,height:38,background:`linear-gradient(135deg,${C.primary},${C.secondary})`,borderRadius:10,display:'flex',alignItems:'center',justifyContent:'center',boxShadow:`0 0 24px rgba(0,229,168,.4),0 0 48px rgba(91,140,255,.2)`,flexShrink:0}},
            h('svg',{width:22,height:22,viewBox:'0 0 24 24',fill:'none',stroke:'white',strokeWidth:1.5,strokeLinecap:'round'},
              h('circle',{cx:12,cy:12,r:8}),
              h('circle',{cx:12,cy:12,r:3,fill:'white',stroke:'none'}),
              ...[0,90,180,270].map(a=>h('line',{key:a,
                x1:12+(a===180?-8:a===0?8:0),y1:12+(a===90?-8:a===270?8:0),
                x2:12+(a===180?-11:a===0?11:0),y2:12+(a===90?-11:a===270?11:0)
              }))
            )
          ),
          h('div',null,
            h('div',{style:{color:'white',fontWeight:700,fontSize:'1rem',letterSpacing:'-.02em'}},
              'AI',h('span',{style:{color:C.primary}},' vs '),'REAL'
            ),
            h('div',{style:{color:'rgba(255,255,255,.3)',fontSize:'.6rem',letterSpacing:'.12em',fontFamily:'monospace'}},'IMAGE DETECTION SYSTEM v2.4.1')
          )
        ),
        h('div',{style:{display:'flex',gap:6,alignItems:'center'}},
          ...['OVERVIEW','HISTORY','API','DOCS'].map(t=>h('button',{key:t,
            style:{background:'none',border:'none',color:'rgba(255,255,255,.3)',cursor:'pointer',fontSize:'.65rem',letterSpacing:'.1em',padding:'6px 10px',fontFamily:'monospace',transition:'color .2s'},
            onMouseEnter:e=>e.target.style.color=C.primary,
            onMouseLeave:e=>e.target.style.color='rgba(255,255,255,.3)'
          },t)),
          h('div',{style:{display:'flex',alignItems:'center',gap:7,marginLeft:8,padding:'6px 12px',border:'1px solid rgba(0,229,168,.12)',borderRadius:6,background:'rgba(0,229,168,.04)'}},
            h('div',{style:{width:7,height:7,borderRadius:'50%',background:C.primary,boxShadow:`0 0 8px ${C.primary}`,animation:'pulse 2s infinite'}}),
            h('span',{style:{color:'rgba(255,255,255,.4)',fontSize:'.6rem',fontFamily:'monospace'}},
              stage==='scanning'?'PROCESSING':'SYSTEM ACTIVE'
            )
          )
        )
      ),

      // ── MAIN ──
      h('main',{style:{flex:1,display:'flex',alignItems:'center',justifyContent:'center',padding:'20px 24px',gap:20}},

        // LEFT PANEL — Stats + Sparkline
        (showStats||showTrend)&&h('div',{style:{width:170,display:'flex',flexDirection:'column',gap:10,flexShrink:0}},
          showStats&&stats.map((s,i)=>h('div',{key:s.l,
            style:{padding:'12px 14px',background:'rgba(255,255,255,.025)',border:'1px solid rgba(255,255,255,.05)',borderRadius:10,animation:`slideLeft .6s ${.3+i*.1}s ease both`,cursor:'default',transition:'border-color .3s,transform .2s'},
            onMouseEnter:e=>{e.currentTarget.style.borderColor='rgba(0,229,168,.2)';e.currentTarget.style.transform='scale(1.02)';},
            onMouseLeave:e=>{e.currentTarget.style.borderColor='rgba(255,255,255,.05)';e.currentTarget.style.transform='scale(1)';}
          },
            h('div',{style:{color:'rgba(255,255,255,.3)',fontSize:'.58rem',letterSpacing:'.12em',fontFamily:'monospace',marginBottom:4}},s.l),
            h('div',{style:{color:s.c||C.primary,fontSize:'1.35rem',fontWeight:700,fontFamily:'monospace',textShadow:`0 0 20px ${(s.c||C.primary)}50`}},s.v)
          )),
          showTrend&&h('div',{style:{padding:'12px',background:'rgba(255,255,255,.015)',border:'1px solid rgba(255,255,255,.04)',borderRadius:10,animation:'slideLeft .6s .7s ease both'}},
            h('div',{style:{color:'rgba(255,255,255,.25)',fontSize:'.58rem',letterSpacing:'.1em',fontFamily:'monospace',marginBottom:8}},'DETECTION TREND'),
            h('svg',{width:'100%',height:44,viewBox:'0 0 140 44'},
              h('defs',null,
                h('linearGradient',{id:'sg',x1:0,y1:0,x2:0,y2:1},
                  h('stop',{offset:'0%',stopColor:C.primary,stopOpacity:.3}),
                  h('stop',{offset:'100%',stopColor:C.primary,stopOpacity:0})
                )
              ),
              h('path',{d:'M0,38 L18,32 L32,18 L48,26 L65,13 L82,20 L96,7 L116,16 L140,4 L140,44 L0,44Z',fill:'url(#sg)'}),
              h('polyline',{points:'0,38 18,32 32,18 48,26 65,13 82,20 96,7 116,16 140,4',fill:'none',stroke:C.primary,strokeWidth:1.5,strokeLinejoin:'round'})
            )
          )
        ),

        // CENTER — Holographic card with tilt
        h('div',{ref:cardRef,onMouseMove:onCardMove,onMouseLeave:onCardLeave,style:{perspective:1200,width:460,flexShrink:0}},
          h('div',{style:{transform:`perspective(1200px) rotateX(${cardRot.x}deg) rotateY(${cardRot.y}deg)`,transformStyle:'preserve-3d',transition:'transform .05s linear'}},
            h('div',{style:{
              background:'rgba(4,6,14,.78)',
              backdropFilter:'blur(40px) saturate(180%)',
              border:'1px solid rgba(0,229,168,.13)',
              borderRadius:18,padding:'26px',
              boxShadow:'0 0 80px rgba(0,229,168,.06),0 0 160px rgba(91,140,255,.04),inset 0 0 80px rgba(0,229,168,.015),0 40px 80px rgba(0,0,0,.6)',
              position:'relative',overflow:'hidden',
              animation:glitch?'glitch .15s infinite':'breathe 4s infinite',
            }},
              // Glitch overlay
              glitch&&h('div',{style:{position:'absolute',inset:0,background:'rgba(255,77,77,.06)',pointerEvents:'none',zIndex:50}}),
              // Shimmer sweep
              h('div',{style:{position:'absolute',top:0,left:0,right:0,bottom:0,overflow:'hidden',pointerEvents:'none',borderRadius:'inherit'}},
                h('div',{style:{position:'absolute',top:0,width:'50%',height:'100%',background:'linear-gradient(90deg,transparent,rgba(255,255,255,.015),transparent)',animation:'shimmer 3s 2s infinite linear',zIndex:1}})
              ),
              // Corner brackets
              ...[
                {t:10,l:10,bT:`1.5px solid ${C.primary}`,bL:`1.5px solid ${C.primary}`},
                {t:10,r:10,bT:`1.5px solid ${C.primary}`,bR:`1.5px solid ${C.primary}`},
                {b:10,l:10,bB:`1.5px solid ${C.primary}`,bL:`1.5px solid ${C.primary}`},
                {b:10,r:10,bB:`1.5px solid ${C.primary}`,bR:`1.5px solid ${C.primary}`}
              ].map((c,i)=>h('div',{key:i,style:{position:'absolute',width:18,height:18,
                ...(c.t!==undefined?{top:c.t}:{}),
                ...(c.b!==undefined?{bottom:c.b}:{}),
                ...(c.l!==undefined?{left:c.l}:{}),
                ...(c.r!==undefined?{right:c.r}:{}),
                borderTop:c.bT,borderLeft:c.bL,borderBottom:c.bB,borderRight:c.bR
              }})),

              // Card header row
              h('div',{style:{display:'flex',alignItems:'center',justifyContent:'space-between',marginBottom:20}},
                h('div',{style:{color:C.primary,fontSize:'.6rem',letterSpacing:'.18em',fontFamily:'monospace'}},'◈ NEURAL DETECTION ENGINE'),
                h('div',{style:{display:'flex',alignItems:'center',gap:6}},
                  h('div',{style:{
                    width:6,height:6,borderRadius:'50%',
                    background:stage==='scanning'?C.primary:stage==='result'?rc:'rgba(255,255,255,.2)',
                    boxShadow:stage!=='idle'?`0 0 8px ${stage==='result'?rc:C.primary}`:'none',
                    animation:stage==='scanning'?'pulse 1.2s infinite':''
                  }}),
                  h('span',{style:{color:'rgba(255,255,255,.35)',fontSize:'.58rem',fontFamily:'monospace',letterSpacing:'.08em'}},
                    stage==='idle'?'STANDBY':stage==='scanning'?'PROCESSING':'COMPLETE'
                  )
                )
              ),

              // ── IDLE STATE ──
              stage==='idle'&&h('div',{key:'idle'},
                h('div',{style:{color:'rgba(255,255,255,.45)',fontSize:'.62rem',letterSpacing:'.2em',fontFamily:'monospace',marginBottom:10}},'UPLOAD IMAGE'),
                h('label',{
                  htmlFor:'fi',
                  onDragOver:e=>{e.preventDefault();setDragging(true);},
                  onDragLeave:()=>setDragging(false),
                  onDrop:e=>{e.preventDefault();setDragging(false);processFile(e.dataTransfer.files[0]);},
                  style:{
                    display:'block',height:260,
                    border:`1px dashed ${dragging?C.primary:'rgba(0,229,168,.25)'}`,
                    borderRadius:10,cursor:'pointer',position:'relative',overflow:'hidden',
                    background:dragging?'rgba(0,229,168,.04)':'rgba(0,229,168,.015)',
                    transition:'border-color .3s,background .3s',
                    boxShadow:dragging?`0 0 30px rgba(0,229,168,.2)`:'none'
                  }
                },
                  h('input',{id:'fi',type:'file',accept:'image/*',onChange:e=>{processFile(e.target.files[0]);e.target.value='';},style:{display:'none'}}),
                  burst&&h('div',{style:{position:'absolute',inset:0,background:`radial-gradient(circle,rgba(0,229,168,.4) 0%,transparent 65%)`,animation:'ripple .9s ease-out forwards',pointerEvents:'none'}}),
                  h('div',{style:{position:'absolute',inset:0,overflow:'hidden',pointerEvents:'none'}},
                    h('div',{style:{position:'absolute',top:0,width:'40%',height:'100%',background:'linear-gradient(90deg,transparent,rgba(0,229,168,.05),transparent)',animation:'shimmer 4s infinite linear'}})
                  ),
                  h(FloatingParticles,{active:dragging}),
                  h('div',{style:{position:'absolute',inset:0,display:'flex',flexDirection:'column',alignItems:'center',justifyContent:'center',gap:14}},
                    h('div',{style:{width:68,height:68,borderRadius:'50%',border:`1px solid rgba(0,229,168,.2)`,display:'flex',alignItems:'center',justifyContent:'center',animation:'float 3s infinite ease-in-out',boxShadow:`0 0 30px rgba(0,229,168,.08)`}},
                      h('svg',{width:30,height:30,viewBox:'0 0 24 24',fill:'none',stroke:C.primary,strokeWidth:1.5,strokeLinecap:'round',strokeLinejoin:'round'},
                        h('path',{d:'M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4'}),
                        h('polyline',{points:'17 8 12 3 7 8'}),
                        h('line',{x1:12,y1:3,x2:12,y2:15})
                      )
                    ),
                    h('div',{style:{textAlign:'center'}},
                      h('div',{style:{color:'rgba(255,255,255,.75)',fontSize:'.9rem',fontWeight:500,marginBottom:5}},
                        dragging?'Release to analyze':'Drop image to analyze'
                      ),
                      h('div',{style:{color:'rgba(255,255,255,.25)',fontSize:'.7rem',fontFamily:'monospace',letterSpacing:'.04em'}},'or click to browse files'),
                      h('button',{type:'button',style:{
                        marginTop:10,
                        padding:'6px 14px',
                        border:'1px solid rgba(255,255,255,.12)',
                        borderRadius:8,
                        background:'rgba(255,255,255,.06)',
                        color:'rgba(255,255,255,.8)',
                        fontSize:'.7rem',
                        fontFamily:'monospace',
                        letterSpacing:'.06em',
                        cursor:'pointer'
                      }},'Browse files')
                    ),
                    h('div',{style:{display:'flex',gap:8}},
                      ...['JPG','PNG','JPEG'].map(f=>h('span',{key:f,style:{padding:'3px 8px',border:'1px solid rgba(255,255,255,.07)',borderRadius:4,fontSize:'.58rem',color:'rgba(255,255,255,.25)',fontFamily:'monospace',letterSpacing:'.1em'}},f))
                    )
                  )
                ),
                uploadHint&&h('div',{style:{
                  marginTop:10,
                  color:'rgba(255,255,255,.3)',
                  fontSize:'.62rem',
                  fontFamily:'monospace',
                  letterSpacing:'.04em'
                }},uploadHint),
                showBadges&&h('div',{style:{marginTop:12,padding:'10px 14px',background:'rgba(255,255,255,.02)',border:'1px solid rgba(255,255,255,.04)',borderRadius:8,display:'flex',gap:12,flexWrap:'wrap'}},
                  badges.map((txt,i)=>h('div',{key:i,style:{display:'flex',alignItems:'center',gap:5,color:'rgba(255,255,255,.28)',fontSize:'.6rem',fontFamily:'monospace'}},
                    txt
                  ))
                )
              ),

              // ── SCANNING STATE ──
              stage==='scanning'&&h('div',{key:'scanning',style:{animation:'fadeIn .3s ease'}},
                h('div',{style:{height:260,borderRadius:10,overflow:'hidden',position:'relative',border:'1px solid rgba(0,229,168,.1)'}},
                  h(ScanCanvas,{imageUrl,onComplete:onScanDone})
                ),
                h('div',{style:{marginTop:14,display:'flex',flexDirection:'column',gap:7}},
                  ...['FREQUENCY DOMAIN ANALYSIS','GAN ARTIFACT DETECTION','METADATA VERIFICATION','NEURAL PATTERN MATCHING'].map((lbl,i)=>h('div',{key:lbl,style:{display:'flex',alignItems:'center',gap:8}},
                    h('div',{style:{flex:1,height:2,background:`linear-gradient(90deg,${C.primary},${C.secondary})`,borderRadius:1,opacity:.7,boxShadow:`0 0 6px ${C.primary}`,animation:`grow .4s ${i*.15}s ease both`}}),
                    h('span',{style:{color:'rgba(0,229,168,.7)',fontSize:'.58rem',fontFamily:'monospace',letterSpacing:'.08em',minWidth:200}},lbl)
                  ))
                )
              ),

              // ── RESULT STATE ──
              stage==='result'&&result&&h('div',{key:'result',style:{animation:'fadeIn .5s ease'}},
                h('div',{style:{height:168,borderRadius:10,overflow:'hidden',position:'relative',marginBottom:16,border:`1px solid ${rc}22`,boxShadow:`0 0 30px ${rc}15`}},
                  h('img',{src:imageUrl,style:{width:'100%',height:'100%',objectFit:'cover'}}),
                  h('div',{style:{position:'absolute',inset:0,background:`linear-gradient(to top,${rc}35 0%,transparent 55%)`}}),
                  heatmapUrl&&h('img',{src:heatmapUrl,style:{position:'absolute',inset:0,width:'100%',height:'100%',objectFit:'cover',opacity:.55,mixBlendMode:'screen',pointerEvents:'none'}}),
                  h('div',{style:{position:'absolute',top:10,right:10,padding:'4px 11px',background:`${rc}18`,border:`1px solid ${rc}55`,borderRadius:6,color:rc,fontSize:'.62rem',fontFamily:'monospace',letterSpacing:'.12em',backdropFilter:'blur(10px)',boxShadow:`0 0 15px ${rc}25`}},
                    result.isReal?'✓ AUTHENTIC IMAGE':'⚠ AI GENERATED'
                  ),
                  h('div',{style:{position:'absolute',bottom:10,left:10,display:'flex',gap:8}},
                    h('div',{style:{padding:'3px 8px',background:'rgba(0,0,0,.5)',border:'1px solid rgba(255,255,255,.1)',borderRadius:4,color:'rgba(255,255,255,.5)',fontSize:'.6rem',fontFamily:'monospace'}},'SOURCE: UPLOAD'),
                    modelName&&h('div',{style:{padding:'3px 8px',background:'rgba(0,0,0,.5)',border:'1px solid rgba(255,255,255,.1)',borderRadius:4,color:'rgba(255,255,255,.5)',fontSize:'.6rem',fontFamily:'monospace'}},'MODEL: '+modelName)
                  )
                ),
                h('div',{style:{display:'flex',gap:18,alignItems:'flex-start'}},
                  h(RadialRing,{value:result.conf,color:rc,size:120}),
                  h('div',{style:{flex:1,paddingTop:2}},
                    h('div',{style:{fontSize:'1.4rem',fontWeight:700,color:rc,textShadow:`0 0 20px ${rc}55`,lineHeight:1,marginBottom:3,animation:glitch?'glitch .15s infinite':'none'}},
                      result.isReal?'REAL IMAGE':'AI GENERATED'
                    ),
                    h('div',{style:{color:'rgba(255,255,255,.3)',fontSize:'.65rem',fontFamily:'monospace',marginBottom:10,letterSpacing:'.03em'}},
                      summaryText
                    ),
                    showMetrics&&metrics.map((item,i)=>h('div',{key:i,style:{display:'flex',justifyContent:'space-between',padding:'4px 0',borderBottom:'1px solid rgba(255,255,255,.04)'}},
                      h('span',{style:{color:'rgba(255,255,255,.28)',fontSize:'.62rem',fontFamily:'monospace'}},item.k),
                      h('span',{style:{color:rc,fontSize:'.62rem',fontFamily:'monospace'}},item.v)
                    ))
                  )
                ),
                !hideResetButton&&h('button',{onClick:reset,
                  style:{marginTop:14,width:'100%',padding:'11px',background:'rgba(0,229,168,.07)',border:`1px solid rgba(0,229,168,.22)`,borderRadius:9,color:C.primary,cursor:'pointer',fontSize:'.7rem',fontFamily:'monospace',letterSpacing:'.15em',transition:'all .3s'},
                  onMouseEnter:e=>{e.target.style.background='rgba(0,229,168,.14)';e.target.style.boxShadow=`0 0 25px rgba(0,229,168,.25)`;},
                  onMouseLeave:e=>{e.target.style.background='rgba(0,229,168,.07)';e.target.style.boxShadow='none';}
                },'↺  ANALYZE ANOTHER IMAGE')
              )
            )
          )
        ),
        // RIGHT PANEL — Recent scans + system status
        (showRecent||showSystem)&&h('div',{style:{width:170,display:'flex',flexDirection:'column',gap:10,flexShrink:0}},
          showRecent&&h('div',{style:{padding:'12px 14px',background:'rgba(255,255,255,.02)',border:'1px solid rgba(255,255,255,.05)',borderRadius:10,animation:'slideRight .6s .3s ease both'}},
            h('div',{style:{color:'rgba(255,255,255,.28)',fontSize:'.58rem',letterSpacing:'.1em',fontFamily:'monospace',marginBottom:8}},'RECENT SCANS'),
            recentScans.map((s,i)=>h('div',{key:i,
              style:{display:'flex',justifyContent:'space-between',alignItems:'center',padding:'4px 0',borderBottom:i<recentScans.length-1?'1px solid rgba(255,255,255,.03)':'none'}
            },
              h('div',{style:{display:'flex',alignItems:'center',gap:5}},
                h('div',{style:{width:5,height:5,borderRadius:'50%',background:s.r?C.primary:C.danger,boxShadow:`0 0 5px ${s.r?C.primary:C.danger}`}}),
                h('span',{style:{color:'rgba(255,255,255,.4)',fontSize:'.62rem',fontFamily:'monospace'}},s.r?'REAL':'AI GEN')
              ),
              h('span',{style:{color:'rgba(255,255,255,.25)',fontSize:'.6rem',fontFamily:'monospace'}},s.c+'%')
            ))
          ),
          showSystem&&systemStatus.map((it,i)=>h('div',{key:it.l,
            style:{padding:'9px 12px',background:'rgba(255,255,255,.02)',border:`1px solid ${it.a?'rgba(0,229,168,.12)':'rgba(255,255,255,.04)'}`,borderRadius:8,display:'flex',justifyContent:'space-between',alignItems:'center',animation:`slideRight .6s ${.4+i*.07}s ease both`,transition:'border-color .5s'}
          },
            h('span',{style:{color:'rgba(255,255,255,.35)',fontSize:'.58rem',fontFamily:'monospace',letterSpacing:'.06em'}},it.l),
            h('div',{style:{display:'flex',alignItems:'center',gap:5}},
              h('div',{style:{width:5,height:5,borderRadius:'50%',background:it.a?C.primary:'rgba(255,255,255,.15)',boxShadow:it.a?`0 0 6px ${C.primary}`:'none',animation:it.a?'pulse 1.8s infinite':''}}),
              h('span',{style:{color:it.a?C.primary:'rgba(255,255,255,.18)',fontSize:'.55rem',fontFamily:'monospace'}},it.s)
            )
          ))
        )
      ),

      // ── FOOTER ──
      showFooter&&h('footer',{style:{padding:'8px 28px',borderTop:'1px solid rgba(0,229,168,.05)',display:'flex',justifyContent:'space-between',alignItems:'center',backdropFilter:'blur(20px)',animation:'fadeUp .8s .5s ease both'}},
        footerItems.length>0?h('div',{style:{display:'flex',gap:20}},
          footerItems.map((t,i)=>h('span',{key:i,style:{color:'rgba(255,255,255,.18)',fontSize:'.58rem',fontFamily:'monospace',letterSpacing:'.07em'}},t))
        ):h('div',null),
        footerNote?h('div',{style:{color:'rgba(255,255,255,.18)',fontSize:'.58rem',fontFamily:'monospace',letterSpacing:'.07em'}},footerNote):h('div',null)
      )
    ),

    // Global keyframe animations
    h('style',null,`
      @keyframes fadeDown{from{opacity:0;transform:translateY(-30px)}to{opacity:1;transform:none}}
      @keyframes fadeUp{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:none}}
      @keyframes fadeIn{from{opacity:0}to{opacity:1}}
      @keyframes slideLeft{from{opacity:0;transform:translateX(-30px)}to{opacity:1;transform:none}}
      @keyframes slideRight{from{opacity:0;transform:translateX(30px)}to{opacity:1;transform:none}}
      @keyframes grow{from{transform:scaleX(0);transform-origin:left}to{transform:scaleX(1)}}
      @keyframes trailFade{0%{opacity:.7;transform:scale(1) translateY(0)}100%{opacity:0;transform:scale(.3) translateY(-30px)}}
    `)
  );
}

ReactDOM.render(React.createElement(App), document.getElementById('root'));
</script>
</body>
</html>
"""


def render_ui(data):
    return UI_TEMPLATE.replace("__INITIAL_DATA__", json.dumps(data))


# -------------------------
# Streamlit Shell
# -------------------------
st.set_page_config(
    page_title="AI vs Real Image Detector",
    page_icon="🖼️",
    layout="wide"
)

st.markdown(
    """
    <style>
      .block-container { padding: 0; }
      iframe { border: none; }
      div[data-testid="stFileUploader"] {
        position: fixed;
        left: 50%;
        top: 50%;
        transform: translate(-50%, -50%);
        width: 460px;
        height: 260px;
        opacity: 0;
        z-index: 9999;
      }
      div[data-testid="stFileUploader"] section { height: 100%; }
      div[data-testid="stFileUploader"] input[type="file"] {
        width: 100%;
        height: 100%;
        cursor: pointer;
      }
      div[data-testid="stButton"] {
        position: fixed;
        left: 50%;
        top: 50%;
        transform: translate(-50%, 320px);
        width: min(460px, 92vw);
        z-index: 9999;
      }
      div[data-testid="stButton"] > button {
        width: 100%;
        padding: 11px;
        background: rgba(0,229,168,.07);
        border: 1px solid rgba(0,229,168,.22);
        border-radius: 9px;
        color: #00E5A8;
        cursor: pointer;
        font-size: .7rem;
        font-family: monospace;
        letter-spacing: .15em;
      }
      div[data-testid="stButton"] > button:hover {
        background: rgba(0,229,168,.14);
        box-shadow: 0 0 25px rgba(0,229,168,.25);
      }
    </style>
    """,
    unsafe_allow_html=True
)

total_samples, train_samples, test_samples = build_dataset_counts()
dataset_stats = []
dataset_summary = ""
if total_samples:
    dataset_stats = [
        {"l": "TOTAL SAMPLES", "v": str(total_samples)},
        {"l": "TRAIN SAMPLES", "v": str(train_samples)},
        {"l": "TEST SAMPLES", "v": str(test_samples)}
    ]
    dataset_summary = (
        f"Dataset split ready: {total_samples} total | "
        f"{train_samples} train | {test_samples} test."
    )

if "ui_data" not in st.session_state:
    st.session_state.ui_data = {
        "stage": "idle",
        "imageUrl": None,
        "result": None,
        "hideResetButton": True,
        "metrics": [],
        "heatmapUrl": None,
        "resultSummary": dataset_summary,
        "modelName": "",
        "uploadHint": UPLOAD_HINT,
        "badges": [],
        "footerItems": [],
        "footerNote": "",
        "stats": dataset_stats,
        "showTrend": False,
        "recentScans": [],
        "showSystem": False,
        "systemStatus": []
    }

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

uploaded_file = None
if st.session_state.ui_data.get("stage") == "idle":
    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "png", "jpeg"],
        label_visibility="collapsed",
        key=f"uploader_{st.session_state.uploader_key}"
    )

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    if not file_bytes:
        st.error("The uploaded file is empty.")
    else:
        max_bytes = MAX_UPLOAD_MB * 1024 * 1024
        if len(file_bytes) > max_bytes:
            st.error(f"File too large. Max size is {MAX_UPLOAD_MB}MB.")
            st.stop()
        file_array = np.asarray(bytearray(file_bytes), dtype=np.uint8)
        image = cv2.imdecode(file_array, 1)
        if image is None:
            st.error("Could not read the image file.")
        elif model is None or scaler is None:
            st.error("Model files not found. Train the model first.")
        else:
            features = extract_features_from_image(image)
            if features is None:
                st.error("Could not extract features from the image.")
            else:
                features_scaled = scaler.transform([features])
                prob_ai, confidence = predict_probabilities(model, features_scaled)
                is_real = prob_ai < 0.5
                conf_pct = int(round(confidence * 100.0))
                ai_pct = int(round(prob_ai * 100.0))
                model_name = type(model).__name__
                if hasattr(model, "kernel"):
                    model_name = f"{model_name} ({model.kernel})"

                metrics_list = [
                    {"k": "AI Probability", "v": f"{ai_pct}%"},
                    {"k": "Confidence", "v": f"{conf_pct}%"}
                ]

                st.session_state.ui_data = {
                    "stage": "result",
                    "imageUrl": image_bytes_to_data_url(file_bytes, uploaded_file.type),
                    "heatmapUrl": None,
                    "result": {
                        "isReal": bool(is_real),
                        "conf": conf_pct
                    },
                    "hideResetButton": True,
                    "metrics": metrics_list,
                    "resultSummary": f"Estimated AI probability: {ai_pct}%.",
                    "modelName": model_name,
                    "uploadHint": UPLOAD_HINT,
                    "badges": [],
                    "footerItems": [],
                    "footerNote": "",
                    "stats": dataset_stats,
                    "showTrend": False,
                    "recentScans": [],
                    "showSystem": False,
                    "systemStatus": []
                }
                st.rerun()

components.html(
    render_ui(st.session_state.ui_data),
    height=960,
    scrolling=False
)

if st.session_state.ui_data.get("stage") == "result":
    if st.button("ANALYZE ANOTHER IMAGE"):
        st.session_state.ui_data = {
            "stage": "idle",
            "imageUrl": None,
            "result": None,
            "hideResetButton": True,
            "metrics": [],
            "heatmapUrl": None,
            "resultSummary": dataset_summary,
            "modelName": "",
            "uploadHint": UPLOAD_HINT,
            "badges": [],
            "footerItems": [],
            "footerNote": "",
            "stats": dataset_stats,
            "showTrend": False,
            "recentScans": [],
            "showSystem": False,
            "systemStatus": []
        }
        st.session_state.uploader_key += 1
        st.rerun()
