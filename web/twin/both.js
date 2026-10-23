/* ---- heat x flood view ---- */
(function(){
"use strict";
const $=s=>document.querySelector(s), F=FLOOD_API, N=F.N, W=BUNDLE.wards, U=F.U;
let sel=-1, fHaz=null, fl=null, hr=null, hp=null, fp=null, tier=null;
const cssv=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
function pct(a){ const idx=[...Array(N).keys()].sort((x,y)=>a[x]-a[y]), p=new Float64Array(N);
  for(let k=0;k<N;k++) p[idx[k]]=100*k/(N-1); return p; }
function rank(a){ const idx=[...Array(N).keys()].sort((x,y)=>a[x]-a[y]), r=new Float64Array(N);
  for(let a0=0;a0<N;){ let b=a0; while(b+1<N&&a[idx[b+1]]===a[idx[a0]]) b++; const v=(a0+b)/2; for(let k=a0;k<=b;k++) r[idx[k]]=v; a0=b+1; } return r; }
function spearman(a,b){ const x=rank(a),y=rank(b),mx=x.reduce((s,v)=>s+v,0)/N,my=y.reduce((s,v)=>s+v,0)/N; let sxy=0,sx=0,sy=0;
  for(let i=0;i<N;i++){ sxy+=(x[i]-mx)*(y[i]-my); sx+=(x[i]-mx)**2; sy+=(y[i]-my)**2; } return sxy/Math.sqrt(sx*sy); }
const fmt=n=>n>=1e6?(n/1e6).toFixed(2)+"m":n>=1e3?(n/1e3).toFixed(0)+"k":Math.round(n);

/* bivariate 3x3: columns = flood tercile, rows = heat tercile */
const BV=[["#ececea","#a9cce3","#4f8fc0"],["#f2c9a0","#b9a6a0","#5f7f9a"],["#e0894f","#a35c5a","#2d3a63"]];
const mix=(h,f)=>BV[h][f];
const terc=p=>p>=66.7?2:p>=33.3?1:0;

function compute(){
  const rs=F.rainState(), res=F.simulate(F.uniform(rs.rain),rs.dur,rs.drain);
  fl=res.units.map(u=>u.risk); fHaz=res.units.map(u=>u.hazard); hr=Array.from({length:N},(_,i)=>HEAT_API.wardRisk(i));
  hp=pct(hr); fp=pct(fl);
  $("#b-storm").textContent=`flood storm: ${rs.rain} mm / ${rs.dur} h / ${rs.drain} mm/h drainage, citywide`;
}
const kit=MapKit($("#b-map"),{
  fill:i=>fl?mix(terc(hp[i]),terc(fp[i])):null
});
function legend(){
  $("#b-leg").innerHTML=`<span style="display:inline-grid;grid-template-columns:repeat(3,16px);gap:2px">${[2,1,0].flatMap(h=>[0,1,2].map(f=>`<i style="width:16px;height:16px;border-radius:2px;border:1px solid var(--rule);background:${mix(h,f)}"></i>`)).join("")}</span>
   <span>→ flood risk higher &nbsp; ↑ heat risk higher &nbsp; (thirds across 198 wards)</span>`;
}
function draw(){ kit.selected=sel; kit.draw(); legend(); }

/* what-if presets: heat levers map onto flood surface changes */
const PRE=[
 {n:"Cool roofs",L:{ndvi:0,tree:0,roof:.10,imp:0,water:0}},
 {n:"Street trees (+15 pts canopy)",L:{ndvi:.05,tree:.15,roof:0,imp:0,water:0}},
 {n:"De-pave 10 pts",L:{ndvi:0,tree:0,roof:0,imp:.10,water:0}},
 {n:"Add water bodies (+2 pts)",L:{ndvi:0,tree:0,roof:0,imp:0,water:.02}},
 {n:"All of the above + greening",L:{ndvi:.20,tree:.15,roof:.10,imp:.10,water:.02}}];
function floodDelta(i,L){
  const rs=F.rainState(), r=F.uniform(rs.rain);
  const base=F.simulate(r,rs.dur,rs.drain).units[i];
  const ov={}; ov[i]={impervious:Math.max(.01,U[i].impervious-L.imp),tree_canopy:Math.min(.7,U[i].tree_canopy+L.tree),water:Math.min(.35,U[i].water+L.water)};
  const s=F.simulate(r,rs.dur,rs.drain,ov).units[i];
  return {hazard:s.hazard-base.hazard, ponding:s.ponding-base.ponding};
}
function panel(){
  if(sel<0){ return; }
  $("#b-name").textContent=W[sel];
  const row=(k,v)=>`<div class="frow"><span>${k}</span><span class="mono">${v}</span></div>`;
  $("#b-detail").innerHTML=row("Heat risk",`${hr[sel].toFixed(0)}  (${Math.round(hp[sel])}th pct)`)+row("Flood risk (storm scenario)",`${fl[sel].toFixed(0)}  (${Math.round(fp[sel])}th pct)`)+
    row("Learned flood susceptibility",`${Math.round(F.SUSC[sel])}th pct`)+row("BBMP-listed points",F.M.listed_points[sel]||"none")+
    row("Land surface temperature",U[sel].lst.toFixed(1)+" °C")+row("Population",fmt(U[sel].population));
  $("#b-int").innerHTML=`<table class="vt"><thead><tr><th>Intervention</th><th>Heat risk Δ</th><th>Flood hazard Δ</th><th>Verdict</th></tr></thead><tbody>`+
   PRE.map(p=>{ const dh=HEAT_API.scenarioDelta(sel,p.L), df=floodDelta(sel,p.L).hazard;
     const cls=d=>d<-0.05?"good":d>0.05?"bad":"", sg=d=>(d>0?"+":d<0?"−":"")+Math.abs(d).toFixed(1);
     const v=dh<-0.05&&df<=0.05?"helps heat"+(df<-0.05?" and flood":", flood unchanged"):dh<=0.05&&df<-0.05?"helps flood, heat unchanged":dh<-0.05&&df>0.05?"trade-off":"little effect";
     return `<tr><td>${p.n}</td><td class="mono ${cls(dh)}">${sg(dh)}</td><td class="mono ${cls(df)}">${sg(df)}</td><td>${v}</td></tr>`; }).join("")+
   `</tbody></table><p class="fnote">Negative is better. Changes apply to this ward's surfaces only. Flood hazard is the cascade's 0–100 hazard under the Flood tab's storm. The heat figure is the learned model's risk change (model MAE bounds how finely it can be read).</p>`;
}
function table(){
  const rows=[...Array(N).keys()].map(i=>({i,w:Math.max(hp[i],fp[i])})).sort((a,b)=>b.w-a.w).slice(0,50);
  $("#b-tbody").innerHTML=rows.map(({i,w})=>`<tr data-i="${i}" aria-selected="${i===sel}"><td><div class="fname" title="${W[i]}">${W[i]}</div></td>
   <td class="mono">${hr[i].toFixed(0)}</td><td class="mono">${Math.round(hp[i])}</td><td class="mono">${fl[i].toFixed(0)}</td><td class="mono">${Math.round(fp[i])}</td><td class="mono">${Math.round(F.SUSC[i])}</td>
   <td>${hp[i]>=fp[i]?"heat":"flood"} ${Math.round(w)}</td></tr>`).join("");
  $("#b-tbody").querySelectorAll("tr").forEach(tr=>tr.onclick=()=>pick(+tr.dataset.i));
}
function kpis(){
  const rho=spearman(hr,fl); $("#b-k-rho").textContent=rho.toFixed(2);
  const rh=spearman(U.map(u=>u.lst),fHaz); $("#b-k-rho-s").textContent=`risk scores · hazards alone (LST vs flood hazard): ${rh.toFixed(2)}`;
  let both=0,h=0,f=0,pop=0;
  for(let i=0;i<N;i++){ const a=hp[i]>=66.7,b=fp[i]>=66.7; if(a&&b){both++;pop+=U[i].population;} else if(a)h++; else if(b)f++; }
  $("#b-k-both").textContent=both; $("#b-k-pop").textContent=fmt(pop); $("#b-k-h").textContent=h; $("#b-k-f").textContent=f;
}
function refresh(){ compute(); kpis(); table(); panel(); draw(); }
function pick(i){ sel=i; TWIN.emit("both",i); panel(); table(); draw(); }
const cv=$("#b-map");
cv.addEventListener("click",e=>{ const i=kit.wardAt(...kit.pos(e)); if(i>=0) pick(i); });
cv.addEventListener("pointermove",e=>{ const i=kit.wardAt(...kit.pos(e));
  if(i!==kit.hover){ kit.hover=i; $("#b-hover").textContent=i>=0?`${W[i]} — heat ${hr[i].toFixed(0)}, flood ${fl[i].toFixed(0)}`:"Hover a ward. Click to select it in every view."; draw(); } });
cv.addEventListener("pointerleave",()=>{kit.hover=-1;draw();});
window.addEventListener("resize",()=>{ if(TWIN.view==="both") draw(); });
new MutationObserver(()=>{ if(TWIN.view==="both") draw(); }).observe(document.documentElement,{attributes:true,attributeFilter:["data-theme"]});
TWIN.register("both",{ selected:()=>sel, select:i=>{sel=i;},
  onShow:()=>{ refresh(); } });
TWIN.on((src,i)=>{ sel=i; });
})();
