/* ---- shell: shared selection bus, view switching, shared ward-polygon map ---- */
"use strict";
window.TWIN = (function(){
  const views=["heat","flood","both"], subs=[], api={};
  let sel=-1, cur="heat";
  const T={
    get sel(){return sel}, get view(){return cur},
    on(fn){subs.push(fn)},
    register(name,a){api[name]=a},
    emit(src,i){ if(i==null||i<0||i===sel) return; sel=i; subs.forEach(f=>f(src,i)); },
    show(v){
      if(!views.includes(v)) v="heat";
      cur=v;
      views.forEach(x=>{ const e=document.getElementById("view-"+x); if(e) e.hidden=(x!==v); });
      document.querySelectorAll("#hazardTabs button").forEach(b=>b.setAttribute("aria-pressed",String(b.dataset.view===v)));
      // bring the newly shown view onto the shared selection, then let it size its canvas
      const a=api[v];
      if(a && sel>=0 && a.selected && a.selected()!==sel) a.select(sel);
      window.dispatchEvent(new Event("resize"));
      if(a && a.onShow) a.onShow();
      try{ history.replaceState(null,"","#"+v) }catch(_){}
    }
  };
  return T;
})();

/* Shared renderer for the ward-polygon maps (flood and heat x flood views). */
window.MapKit = function(canvas, opts){
  const ctx=canvas.getContext("2d"), WG=BUNDLE.wardGeo, ZS=BUNDLE.zoneSegs||[], N=WG.length;
  let x0=1e9,y0=1e9,x1=-1e9,y1=-1e9;
  const WBB=WG.map(rs=>{ let a=1e9,b=1e9,c=-1e9,d=-1e9;
    for(const r of rs) for(let k=0;k<r.length;k+=2){
      if(r[k]<a)a=r[k]; if(r[k]>c)c=r[k]; if(r[k+1]<b)b=r[k+1]; if(r[k+1]>d)d=r[k+1]; }
    if(a<x0)x0=a; if(c>x1)x1=c; if(b<y0)y0=b; if(d>y1)y1=d; return [a,b,c,d]; });
  const KX=Math.cos((y0+y1)/2*Math.PI/180);
  const kit={ctx,N,WBB,view:{s:1,ox:0,oy:0,w:900,h:600},hover:-1,selected:-1};
  const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  kit.css=css;
  kit.layout=function(){
    const cssW=canvas.parentElement.clientWidth||900;
    const gw=(x1-x0)*KX, gh=(y1-y0);
    const cssH=Math.max(340,Math.min(640,cssW*(gh/gw)));
    const dpr=Math.min(2,window.devicePixelRatio||1);
    canvas.width=cssW*dpr; canvas.height=cssH*dpr; canvas.style.height=cssH+"px";
    ctx.setTransform(dpr,0,0,dpr,0,0);
    const pad=10, s=Math.min((cssW-2*pad)/gw,(cssH-2*pad)/gh);
    kit.view={s,ox:(cssW-gw*s)/2,oy:(cssH-gh*s)/2,w:cssW,h:cssH};
  };
  kit.px=lon=>kit.view.ox+(lon-x0)*KX*kit.view.s;
  kit.py=lat=>kit.view.oy+(y1-lat)*kit.view.s;
  function pip(x,y,r){ let ins=false;
    for(let i=0,j=r.length-2;i<r.length;j=i,i+=2){
      const xi=r[i],yi=r[i+1],xj=r[j],yj=r[j+1];
      if((yi>y)!==(yj>y) && x<(xj-xi)*(y-yi)/(yj-yi)+xi) ins=!ins; }
    return ins; }
  kit.wardAt=function(cx,cy){
    const V=kit.view, lon=x0+(cx-V.ox)/(V.s*KX), lat=y1-(cy-V.oy)/V.s;
    for(let i=0;i<N;i++){ const b=WBB[i];
      if(lon<b[0]||lon>b[2]||lat<b[1]||lat>b[3]) continue;
      let ins=false; for(const r of WG[i]) if(pip(lon,lat,r)) ins=!ins;
      if(ins) return i; }
    return -1; };
  kit.pos=function(e){ const r=canvas.getBoundingClientRect(); return [e.clientX-r.left,e.clientY-r.top]; };
  kit.path=function(i){ ctx.beginPath();
    for(const r of WG[i]){ ctx.moveTo(kit.px(r[0]),kit.py(r[1]));
      for(let k=2;k<r.length;k+=2) ctx.lineTo(kit.px(r[k]),kit.py(r[k+1])); ctx.closePath(); } };
  kit.draw=function(){
    kit.layout();
    const base=css("--panel2"), rule=css("--rule"), ink=css("--ink");
    ctx.clearRect(0,0,kit.view.w,kit.view.h);
    ctx.lineJoin="round";
    for(let i=0;i<N;i++){
      kit.path(i);
      ctx.fillStyle=(opts.fill&&opts.fill(i))||base; ctx.fill();
      ctx.strokeStyle=rule; ctx.lineWidth=0.5; ctx.stroke();
      if(opts.hatch&&opts.hatch(i)){
        ctx.save(); ctx.clip(); ctx.strokeStyle="rgba(31,111,156,.5)"; ctx.lineWidth=1;
        const b=WBB[i], ax=kit.px(b[0]), bx=kit.px(b[2]), ay=kit.py(b[3]), by=kit.py(b[1]);
        for(let x=ax-(by-ay);x<bx;x+=6){ ctx.beginPath(); ctx.moveTo(x,ay); ctx.lineTo(x+(by-ay),by); ctx.stroke(); }
        ctx.restore(); }
    }
    // zone borders: zoneSegs is a flat array, 4 numbers (lon1,lat1,lon2,lat2) per segment
    ctx.strokeStyle=rule; ctx.lineWidth=1.1; ctx.beginPath();
    for(let k=0;k+3<ZS.length;k+=4){ ctx.moveTo(kit.px(ZS[k]),kit.py(ZS[k+1])); ctx.lineTo(kit.px(ZS[k+2]),kit.py(ZS[k+3])); }
    ctx.stroke();
    if(opts.extra) opts.extra(kit);
    for(const [i,lw] of [[kit.hover,1.6],[kit.selected,2.4]]){
      if(i<0) continue; kit.path(i); ctx.strokeStyle=ink; ctx.lineWidth=lw; ctx.stroke(); }
  };
  return kit;
};

(function(){
  document.querySelectorAll("#hazardTabs button").forEach(b=>b.onclick=()=>TWIN.show(b.dataset.view));
  window.addEventListener("DOMContentLoaded",()=>{});
})();
