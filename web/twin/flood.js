/* ---- flood view: cascade simulator + learned susceptibility + BBMP validation ---- */
(function(){
"use strict";
const $=s=>document.querySelector(s);
const U=FLOOD.units, T=FLOOD.topology, CFG=FLOOD.floodConfig, M=FLOOD.model, W=BUNDLE.wards, N=W.length;
const PTS=FLOOD.points;                       // [lon,lat,layerCode] 0 vulnerable 1 prone 2 low-lying
const clamp01=x=>Math.min(1,Math.max(0,x));

/* ----- cascade: port of pipeline/flood_model.py. `ov` = {ward: {impervious,tree_canopy,water}} what-if overrides ----- */
function curveNumber(imp,can,wat){
  imp=clamp01(imp);can=clamp01(can);wat=clamp01(wat);
  const tot=imp+can+wat; if(tot>1){imp/=tot;can/=tot;wat/=tot;}
  const per=Math.max(0,1-imp-can-wat);
  return Math.min(100,Math.max(30,imp*CFG.cn_impervious+can*CFG.cn_canopy+wat*CFG.cn_water+per*CFG.cn_pervious));
}
function runoffDepth(P,cn){
  if(P<=0) return 0;
  const S=25400/cn-254, Ia=CFG.ia_ratio*S;
  return P<=Ia?0:(P-Ia)*(P-Ia)/(P-Ia+S);
}
const OUTS=Array.from({length:N},()=>[]);
for(const [u,v,w] of T.edges) OUTS[u].push([v,w]);
const DROP=OUTS.map(o=>o.length?o.reduce((a,[,w])=>a+w,0)/o.length:0);
const DMAX=Math.max(...DROP)||1, SLOPE=DROP.map(d=>Math.min(1,d/DMAX));
const ORDER=[...Array(N).keys()].sort((a,b)=>T.elevation[b]-T.elevation[a]);
const LOGD=U.map(u=>Math.log1p(u.pop_density));
const LDMIN=Math.min(...LOGD), LDMAX=Math.max(...LOGD);
const expo=i=>clamp01((LOGD[i]-LDMIN)/(LDMAX-LDMIN))*100;
const water=(i,ov)=>ov&&ov[i]&&ov[i].water!=null?ov[i].water:U[i].water;
function retention(i,ov){
  return Math.min(.95,Math.max(.05,CFG.retention_base+CFG.retention_water_gain*clamp01(water(i,ov))-CFG.retention_slope_penalty*SLOPE[i]));
}
function simulate(rain,dur,drain,ov){
  const lm=new Float64Array(N), lv=new Float64Array(N), cnA=new Float64Array(N);
  for(let i=0;i<N;i++){
    const o=ov&&ov[i], imp=o&&o.impervious!=null?o.impervious:U[i].impervious, can=o&&o.tree_canopy!=null?o.tree_canopy:U[i].tree_canopy;
    cnA[i]=curveNumber(imp,can,water(i,ov));
    lm[i]=runoffDepth(rain[i]||0,cnA[i]); lv[i]=lm[i]*T.area_km2[i]*1000;
  }
  const inflow=new Float64Array(N), ret=new Float64Array(N), out=new Float64Array(N); let exported=0;
  for(const i of ORDER){
    const through=lv[i]+inflow[i]; if(through<=0) continue;
    if(!OUTS[i].length){ const l=through*CFG.sink_export; exported+=l; ret[i]=through-l; out[i]=l; continue; }
    ret[i]=through*retention(i,ov); const pass=through-ret[i]; out[i]=pass;
    let ws=0; for(const [,w] of OUTS[i]) ws+=w; if(!ws) ws=1;
    for(const [v,w] of OUTS[i]) inflow[v]+=pass*(w/ws);
  }
  const units=[]; let ponded=0, tr=0, tro=0;
  for(let i=0;i<N;i++){
    const aM=T.area_km2[i]*1000, raw=aM>0?ret[i]/aM:0;
    const bu=clamp01(U[i].builtup);
    const cap=drain*dur*(1-CFG.drainage_builtup_penalty*bu);
    const depth=Math.max(0,raw-cap); ponded+=depth*aM;
    const hazard=depth>0?Math.min(100,100*Math.sqrt(depth/CFG.depth_hi_mm)):0;
    const through=lv[i]+inflow[i];
    units.push({i,ward:W[i],rain:rain[i]||0,cn:cnA[i],runoffMm:lm[i],inflow:inflow[i],outflow:out[i],ponding:depth,
      hazard,isSink:!OUTS[i].length,upstream:through>0?inflow[i]/through:0,
      risk:CFG.w_hazard*hazard+CFG.w_exposure*expo(i)+CFG.w_vulnerability*U[i].vulnerability*100});
    tr+=(rain[i]||0)*aM; tro+=lv[i];
  }
  return {units,totalRain:tr,totalRunoff:tro,exported,ponded,drained:Math.max(0,tro-exported-ponded)};
}
const uniform=r=>new Float64Array(N).fill(r);
const band=x=>{let k=0;for(const b of CFG.bands) if(x>=b)k++;return CFG.band_names[k];};

/* ----- validation helpers ----- */
function auc(score,pos){            // Mann-Whitney with average ranks
  const idx=[...Array(N).keys()].sort((a,b)=>score[a]-score[b]), rank=new Float64Array(N);
  for(let a=0;a<N;){ let b=a; while(b+1<N&&score[idx[b+1]]===score[idx[a]]) b++;
    const r=(a+b)/2+1; for(let k=a;k<=b;k++) rank[idx[k]]=r; a=b+1; }
  let np=0,sr=0; for(let i=0;i<N;i++) if(pos[i]){np++;sr+=rank[i];}
  const nn=N-np; return np&&nn?(sr-np*(np+1)/2)/(np*nn):NaN;
}
const FLAG=M.listed_points.map(v=>v>0?1:0), NFLAG=FLAG.reduce((a,b)=>a+b,0);
const AREA=Array.from(T.area_km2);
const SUSC=M.susc_percentile_oof;
const REF=M.ref_storm;

/* ----- state ----- */
let rain=new Float64Array(N), result=null, layer="hazard", sortKey="hazard", brush="ward", sel=-1, showPts=false, painting=false, erasing=false;
const fmt=n=>n>=1e9?(n/1e9).toFixed(2)+"bn":n>=1e6?(n/1e6).toFixed(1)+"m":n>=1e3?(n/1e3).toFixed(0)+"k":Math.round(n);
const fmtM3=n=>n>=1e6?(n/1e6).toFixed(1)+" Mm³":fmt(n)+" m³";
const lerp=(a,b,t)=>a+(b-a)*t;
function ramp(st,t){t=clamp01(t);const n=st.length-1,k=Math.min(n-1,Math.floor(t*n)),f=t*n-k,a=st[k],b=st[k+1];
  return `rgb(${Math.round(lerp(a[0],b[0],f))},${Math.round(lerp(a[1],b[1],f))},${Math.round(lerp(a[2],b[2],f))})`;}
const WATER=[[238,244,246],[190,219,232],[124,183,214],[57,134,184],[22,84,132],[12,44,78]];
const HEAT=[[247,244,233],[247,215,163],[240,168,110],[221,110,74],[176,58,52],[110,27,38]];
const RAINR=[[240,244,242],[203,228,240],[142,199,228],[70,157,206],[28,102,158]];
const SUSCR=[[245,242,236],[230,213,184],[205,160,110],[160,100,70],[110,56,48],[70,30,40]];
const EVC={both:"#B2482C",cas:"#2E79A8",lrn:"#C8873A",none:null};
const evClass=i=>{ if(!result) return "none"; const c=result.units[i].hazard>=50, l=SUSC[i]>=75; return c&&l?"both":c?"cas":l?"lrn":"none"; };
const cssv=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();

function fillFor(i){
  const u=result&&result.units[i];
  if(layer==="susc") return ramp(SUSCR,SUSC[i]/100);
  if(layer==="heat") return ramp(HEAT,(U[i].lst-30)/18);
  if(layer==="evidence"){ const c=EVC[evClass(i)]; return c||null; }
  if(!u) return null;
  if(layer==="rain") return u.rain>0?ramp(RAINR,.25+.75*clamp01(u.rain/300)):null;
  const t=layer==="hazard"?u.hazard/100:layer==="ponding"?Math.min(1,u.ponding/300):u.upstream;
  return t>.001?ramp(WATER,t):null;
}
let wet=0;
const kit=MapKit($("#f-map"),{
  fill:fillFor,
  hatch:i=>wet>0&&wet<N*.75&&layer!=="rain"&&rain[i]>0,
  extra:k=>{
    if(sel>=0&&result){            // upstream contributors of the selected ward
      const up=new Set(); let fr=[sel];
      for(let s=0;s<40&&fr.length;s++){ const nx=[];
        for(const v of fr) for(let u=0;u<N;u++) if(!up.has(u)&&u!==sel&&OUTS[u].some(([w])=>w===v)){up.add(u);nx.push(u);}
        fr=nx; }
      k.ctx.strokeStyle=cssv("--accent"); k.ctx.lineWidth=1.2;
      for(const i of up){ k.path(i); k.ctx.stroke(); }
    }
    if(showPts){
      const C=["#B2482C","#C8873A","#2E79A8"];
      for(let c=2;c>=0;c--){ k.ctx.fillStyle=C[c]; k.ctx.strokeStyle="rgba(255,255,255,.85)"; k.ctx.lineWidth=.8;
        for(const p of PTS){ if(p[2]!==c) continue; k.ctx.beginPath(); k.ctx.arc(k.px(p[0]),k.py(p[1]),c===2?2.6:3.1,0,6.283); k.ctx.fill(); k.ctx.stroke(); } }
    }
  }});
function draw(){ kit.selected=sel; wet=0; for(let i=0;i<N;i++) if(rain[i]>0) wet++; kit.draw(); legend(); }
function legend(){
  const el=$("#f-leg");
  const rp=(st,a,b)=>`<span>${a}</span><span class="framp" style="background:linear-gradient(90deg,${st.map((c,k)=>`rgb(${c[0]},${c[1]},${c[2]}) ${k/(st.length-1)*100}%`).join(",")})"></span><span>${b}</span>`;
  let h="";
  if(layer==="hazard") h=rp(WATER,"no ponding","hazard 100");
  else if(layer==="ponding") h=rp(WATER,"0 mm","300 mm+");
  else if(layer==="upstream") h=rp(WATER,"all local rain","all from upstream");
  else if(layer==="rain") h=rp(RAINR,"dry","300 mm");
  else if(layer==="heat") h=rp(HEAT,"30 °C","48 °C");
  else if(layer==="susc") h=rp(SUSCR,"low","high")+`<span>percentile of learned susceptibility (out-of-fold, independent of the storm)</span>`;
  else h=`<span class="sw"><i style="background:${EVC.both}"></i>both flag it</span><span class="sw"><i style="background:${EVC.cas}"></i>cascade only</span><span class="sw"><i style="background:${EVC.lrn}"></i>learned only</span><span>hazard ≥ 50 and susceptibility pct ≥ 75</span>`;
  if(showPts) h+=`<span class="sw"><i style="background:#B2482C"></i>vulnerable</span><span class="sw"><i style="background:#C8873A"></i>prone</span><span class="sw"><i style="background:#2E79A8"></i>low-lying</span>`;
  el.innerHTML=h;
}

/* ----- run / render ----- */
function run(){
  const dur=+$("#f-s-dur").value, dr=+$("#f-s-drain").value;
  result=rain.some(v=>v>0)?simulate(rain,dur,dr):null;
  kpis(); table(); balance(); detail(); valid(); draw();
}
function kpis(){
  const set=(id,v,s)=>{$("#f-k-"+id).textContent=v; if(s!=null) $("#f-k-"+id+"-s").textContent=s;};
  if(!result){ set("rain","0","no wards selected"); set("runoff","0","—"); set("ret","0","—"); set("wards","0","of "+N); $("#f-k-pop").textContent="0"; return; }
  const r=result, wetN=r.units.filter(u=>u.rain>0).length;
  set("rain",fmtM3(r.totalRain),wetN+" ward"+(wetN===1?"":"s")+" raining");
  set("runoff",fmtM3(r.totalRunoff),r.totalRain>0?Math.round(100*r.totalRunoff/r.totalRain)+"% of rainfall":"—");
  set("ret",fmtM3(r.ponded),r.totalRunoff>0?Math.round(100*r.ponded/r.totalRunoff)+"% of runoff stays":"—");
  set("wards",r.units.filter(u=>u.ponding>0).length,"of "+N+" wards");
  $("#f-k-pop").textContent=fmt(r.units.filter(u=>u.risk>=60).reduce((a,u)=>a+U[u.i].population,0));
}
function balance(){
  const el=$("#f-bal");
  if(!result||result.totalRunoff<=0){ el.innerHTML='<i style="flex:1;background:var(--rule2)"></i>'; return; }
  const r=result,tot=r.totalRunoff;
  el.innerHTML=[[r.drained,"var(--neg)"],[r.ponded,"var(--rain)"],[r.exported,"var(--ink3)"]].map(([v,c])=>`<i style="flex:${Math.max(.001,v/tot)};background:${c}"></i>`).join("");
  $("#f-bk1").textContent="Drained away  "+fmtM3(r.drained); $("#f-bk2").textContent="Standing  "+fmtM3(r.ponded); $("#f-bk3").textContent="Left the city  "+fmtM3(r.exported);
  $("#f-balnote").textContent=`Of ${fmtM3(r.totalRain)} falling, ${fmtM3(r.totalRain-r.totalRunoff)} soaks in or is intercepted. The rest is tracked ward by ward and the three figures sum to it exactly.`;
}
function table(){
  const tb=$("#f-tbody"), em=$("#f-tempty");
  if(!result){ tb.innerHTML=""; em.hidden=false; em.textContent="Nothing is raining yet. Pick a preset or paint on the map."; return; }
  const key={hazard:u=>u.hazard,ponding:u=>u.ponding,upstream:u=>u.upstream,risk:u=>u.risk,susc:u=>SUSC[u.i]}[sortKey];
  const rows=result.units.filter(u=>u.ponding>0||u.rain>0).sort((a,b)=>key(b)-key(a)).slice(0,60);
  if(!rows.length){ tb.innerHTML=""; em.hidden=false; em.textContent="This storm produces no ponding anywhere. Try more rainfall, a shorter storm, or less drainage."; return; }
  em.hidden=true;
  tb.innerHTML=rows.map(u=>`<tr data-i="${u.i}" aria-selected="${u.i===sel}"><td><div class="fname" title="${u.ward}">${u.ward}</div>${u.isSink?'<span class="ctag">outlet</span>':""}</td>
    <td class="mono">${u.rain?Math.round(u.rain):"—"}</td><td class="mono">${u.runoffMm.toFixed(0)}</td><td class="mono">${Math.round(u.upstream*100)}%</td>
    <td class="mono">${u.ponding.toFixed(0)}</td><td class="mono">${u.risk.toFixed(0)}</td><td class="mono">${Math.round(SUSC[u.i])}</td><td class="mono">${M.listed_points[u.i]||"·"}</td></tr>`).join("");
  tb.querySelectorAll("tr").forEach(tr=>tr.onclick=()=>pick(+tr.dataset.i));
}
function detail(){
  const card=$("#f-detailcard");
  if(sel<0){ card.hidden=true; return; }
  card.hidden=false; $("#f-d-name").textContent=W[sel];
  const u=result&&result.units[sel], f=U[sel];
  const row=(k,v)=>`<div class="frow"><span>${k}</span><span class="mono">${v}</span></div>`;
  const heatRisk=window.HEAT_API?HEAT_API.wardRisk(sel):null;
  let h="";
  if(u){
    const up=[]; for(let i=0;i<N;i++) if(OUTS[i].some(([v])=>v===sel)) up.push(W[i]);
    h+=row("Rain applied",u.rain?u.rain.toFixed(0)+" mm":"none")+row("Curve number",u.cn.toFixed(1))+row("Runs off locally",u.runoffMm.toFixed(1)+" mm")+
       row("Arrives from upstream",fmtM3(u.inflow))+row("Passes downstream",fmtM3(u.outflow))+row("Ponding index",u.ponding.toFixed(0)+" mm")+
       row("Cascade flood risk",u.risk.toFixed(0)+"  "+band(u.risk));
    h+=`<p class="fnote">${u.upstream>.5?`${Math.round(u.upstream*100)}% of the water here arrived from upstream (${up.slice(0,3).join(", ")}${up.length>3?" and others":""}).`:`Mostly this ward's own rainfall.${up.length?" Uphill neighbours: "+up.slice(0,3).join(", ")+".":""}`}</p>`;
    if(u.isSink) h+=`<p class="fnote"><b>Domain outlet</b>: nothing downhill is modelled, so ${Math.round(CFG.sink_export*100)}% of arriving water is assumed to leave the city.</p>`;
  }
  h+=row("Learned susceptibility",`${Math.round(SUSC[sel])}th percentile`)+row("BBMP-listed points",`${M.listed_points[sel]||"none"}${M.lowlying[sel]?" · low-lying listed":""}`)+
     (heatRisk!=null?row("Heat risk (for comparison)",heatRisk.toFixed(0)):"")+row("Population",fmt(f.population))+row("Elevation",T.elevation[sel].toFixed(0)+" m");
  $("#f-d-body").innerHTML=h;
}
function modelCard(){
  const m=M.cv.metrics, f=x=>x==null?"—":x.toFixed(3);
  const names=["null (area only)","catchment only","cascade only","static only","full (ridge)","full (boosted trees)"];
  $("#f-model").innerHTML=
   `<p class="fp"><b>What it is.</b> A Poisson ridge regression for the <i>density</i> of BBMP-flagged flood locations in a ward (count per km², ward area as weight). Inputs: ${M.features.length} ward features, including the cascade's own ponding and upstream share at a reference storm (${REF.rain} mm over ${REF.duration} h, ${REF.drainage} mm/h drainage). ${M.n_points_unique} unique points from ${M.n_points_raw} listed.</p>
    <div class="fscroll" style="max-height:none"><table class="vt"><thead><tr><th>Model</th><th>D² vs area-only</th><th>AUC, flagged</th><th>AUC, low-lying*</th></tr></thead><tbody>
    ${names.map(n=>`<tr class="${n==="full (ridge)"?"me":""}"><td>${n}</td><td class="mono">${f(m[n].d2_vs_null.mean)}</td><td class="mono">${f(m[n].auc_flagged.mean)}</td><td class="mono">${f(m[n].auc_lowlying_heldout.mean)}</td></tr>`).join("")}
    </tbody></table></div>
    <p class="fp" style="margin-top:9px">${M.cv.scheme}. Spatial blocks, so a ward is never scored by a model that saw its neighbours. *Low-lying layer never used for training.</p>
    <p class="fp"><b>Honest reading.</b> The static ward features explain real variation in how densely BBMP listed points (D² ≈ ${f(m["static only"].d2_vs_null.mean)}); adding the cascade lifts it only a little (${f(m["full (ridge)"].d2_vs_null.mean)}). Ranking wards as flagged or not is weak: AUC ${f(m["full (ridge)"].auc_flagged.mean)} against ${f(m["null (area only)"].auc_flagged.mean)} for ward area alone. This layer does <b>not</b> change with rainfall.</p>`;
}
function valid(){
  const ref=simulate(uniform(REF.rain),REF.duration,REF.drainage), cur=simulate(uniform(+$("#f-s-rain").value),+$("#f-s-dur").value,+$("#f-s-drain").value);
  const a1=auc(ref.units.map(u=>u.ponding),FLAG), a2=auc(cur.units.map(u=>u.ponding),FLAG), a0=auc(AREA,FLAG), a3=auc(SUSC.map(x=>x),FLAG);
  $("#f-valid").innerHTML=
   `<p class="fp">Ranking all ${N} wards by raw cascade ponding under a <b>citywide</b> storm, how often does a BBMP-flagged ward (${NFLAG} of ${N}) outrank an unflagged one? 0.5 is a coin flip.</p>
    <table class="vt"><tbody>
    <tr><td>Cascade ponding, reference storm (${REF.rain} mm / ${REF.duration} h / ${REF.drainage} mm/h)</td><td class="mono">${a1.toFixed(3)}</td></tr>
    <tr class="me"><td>Cascade ponding, current sliders, citywide</td><td class="mono">${a2.toFixed(3)}</td></tr>
    <tr><td>Ward area alone (bigger wards get more points)</td><td class="mono">${a0.toFixed(3)}</td></tr>
    <tr><td>Learned susceptibility, out-of-fold</td><td class="mono">${a3.toFixed(3)}</td></tr></tbody></table>
    <p class="fp" style="margin-top:9px">Move the drainage slider and watch the second row: the score depends on an assumption nobody has measured. These are in-sample for the cascade (it has no fitted parameters) and out-of-fold for the learned layer. Whatever the figures show, none of them is strong evidence that the model locates flooding.</p>`;
}
function cands(){
  $("#f-cands").innerHTML=`<p class="fp">Wards with high out-of-fold susceptibility but <b>no</b> listed BBMP point. They are candidates for a survey, not predicted floods: the list is incomplete, so some may simply be unsurveyed.</p>`+
   M.unlisted_candidates.map(c=>{const i=W.indexOf(c.ward);return `<div class="cand" data-i="${i}"><span>${c.ward}</span><span class="mono">${Math.round(c.percentile)}th pct</span></div>`;}).join("");
  $("#f-cands").querySelectorAll(".cand").forEach(e=>e.onclick=()=>pick(+e.dataset.i));
}

/* ----- interaction ----- */
function pick(i){ if(i<0) return; sel=i; TWIN.emit("flood",i); table(); detail(); draw(); }
function applyBrush(i,erase){
  const amt=erase?0:+$("#f-s-rain").value; let t=[i];
  if(brush==="neigh") t=[i,...T.neighbours[i]];
  if(brush==="zone"){ const z=BUNDLE.zoneOf[i]; t=[]; for(let k=0;k<N;k++) if(BUNDLE.zoneOf[k]===z) t.push(k); }
  let ch=false; for(const k of t) if(rain[k]!==amt){rain[k]=amt;ch=true;} return ch;
}
const cv=$("#f-map");
cv.addEventListener("pointerdown",e=>{ cv.setPointerCapture(e.pointerId); painting=true; erasing=e.altKey;
  const i=kit.wardAt(...kit.pos(e)); if(i>=0){ sel=i; TWIN.emit("flood",i); if(applyBrush(i,erasing)) run(); else{detail();table();draw();} } });
cv.addEventListener("pointermove",e=>{
  const i=kit.wardAt(...kit.pos(e));
  if(i!==kit.hover){ kit.hover=i;
    $("#f-hover").textContent=i>=0?W[i]+(result&&result.units[i].ponding>0?` — ponding ${result.units[i].ponding.toFixed(0)} mm`:"")+` — susceptibility ${Math.round(SUSC[i])}th pct`:"Click or drag across the map to make it rain";
    if(!painting) draw(); }
  if(painting&&i>=0&&applyBrush(i,erasing)) run(); });
const stop=()=>{painting=false;};
cv.addEventListener("pointerup",stop); cv.addEventListener("pointercancel",stop);
cv.addEventListener("pointerleave",()=>{kit.hover=-1;$("#f-hover").textContent="Click or drag across the map to make it rain";draw();});
function seg(id,attr,fn){ $(id).querySelectorAll("button").forEach(b=>b.onclick=()=>{ $(id).querySelectorAll("button").forEach(x=>x.setAttribute("aria-pressed",String(x===b))); fn(b.dataset[attr]); }); }
seg("#f-layers","layer",v=>{layer=v;draw();}); seg("#f-sortseg","sort",v=>{sortKey=v;table();}); seg("#f-brushseg","brush",v=>{brush=v;});
$("#f-pts").onchange=e=>{showPts=e.target.checked;draw();};
for(const [id,out,after] of [["#f-s-rain","#f-v-rain",()=>{const a=+$("#f-s-rain").value;let any=false;for(let i=0;i<N;i++) if(rain[i]>0){rain[i]=a;any=true;} if(any) run(); else valid();}],["#f-s-dur","#f-v-dur",run],["#f-s-drain","#f-v-drain",run]])
  $(id).addEventListener("input",()=>{$(out).textContent=$(id).value;after();});
$("#f-clear").onclick=()=>{rain.fill(0);run();};
$("#f-d-close").onclick=()=>{sel=-1;table();detail();draw();};
function setRain(pred){ const a=+$("#f-s-rain").value; rain.fill(0); for(let i=0;i<N;i++) if(pred(i)) rain[i]=a; run(); }
$("#f-p-city").onclick=()=>setRain(()=>true);
const lonOf=i=>(kit.WBB[i][0]+kit.WBB[i][2])/2, lons=[...Array(N).keys()].map(lonOf).sort((a,b)=>a-b), LONE=lons[Math.floor(N*.6)];
$("#f-p-east").onclick=()=>setRain(i=>lonOf(i)>=LONE);
$("#f-p-valley").onclick=()=>setRain(i=>T.elevation[i]<=890);
addEventListener("resize",()=>{ if(TWIN.view==="flood") draw(); });

window.FLOOD_API={ simulate, uniform, N, U, CFG, M, SUSC, auc, FLAG, ref:()=>simulate(uniform(REF.rain),REF.duration,REF.drainage),
  rainState:()=>({rain:+$("#f-s-rain").value,dur:+$("#f-s-dur").value,drain:+$("#f-s-drain").value}) };
TWIN.register("flood",{ selected:()=>sel, select:i=>{sel=i;table();detail();draw();}, onShow:()=>{draw();} });
TWIN.on((src,i)=>{ if(src!=="flood"){ sel=i; table(); detail(); if(TWIN.view==="flood") draw(); } });
modelCard(); cands(); setRain(()=>true);
})();
