(async function(){
  async function jget(url){
    const r = await fetch(url, {cache:"no-store"});
    if(!r.ok) throw new Error("HTTP "+r.status);
    return r.json();
  }
  function set(id, v){
    const el = document.getElementById(id);
    if(el) el.textContent = (v===null || v===undefined) ? "—" : String(v);
  }

  try{
    const cams = await jget("/api/cams");
    set("kpiCams", Array.isArray(cams) ? cams.length : "—");
  }catch{}

  try{
    const s = await jget("/api/status");
    if(s && s.ok && s.paused === true) set("kpiMode", "Pause nuit");
    else if(s && s.ok && s.paused === false) set("kpiMode", "Actif");
    else set("kpiMode", "—");
  }catch{}

  try{
    const d = new Date();
    const yyyy = d.getFullYear();
    const mm = String(d.getMonth()+1).padStart(2,"0");
    const dd = String(d.getDate()).padStart(2,"0");
    const today = `${yyyy}-${mm}-${dd}`;

    const st = await jget(`/api/stats?bucket=10m&date=${encodeURIComponent(today)}`);
    set("kpiEvents", st?.totals?.events ?? 0);
    set("kpiVehicles", st?.totals?.vehicles ?? 0);
  }catch{}
})();
