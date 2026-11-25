import re
import os
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..") + "/"
def page(name,title,sub,body,script="",data=False):
    body=re.sub(r"<div class='(read|eq)'>.*?</div>",lambda m:m.group(0).replace(chr(92)+'n',chr(10)),body,flags=re.S)
    d='<script src="assets/run.js"></script>' if data else ""
    open(OUT+name+".html","w").write(f'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><link rel="stylesheet" href="assets/course.css"></head><body><main>
<h1>{title}</h1><p class="sub">{sub}</p><nav id="nav"></nav>
{body}
<div class="pn" id="pn"></div></main>
{d}<script src="assets/course.js"></script>
<script>
nav('{name}');
{script.strip()}
</script></body></html>
''')
def sec(title,tag,inner): return f'<section><h2>{title}<span class="sl">{tag}</span></h2>\n{inner}\n</section>\n'
def split(l,r,even=False): return f'<div class="split{" even" if even else ""}"><div>{l}</div><div>{r}</div></div>'
def key(t): return f'<div class="key"><b>To remember.</b> {t}</div>'
def why(title,t): return f'<div class="why"><b>Why {title}?</b> {t}</div>'
def deep(s,t): return f'<details><summary>Deep dive: {s}</summary>{t}</details>'
def anchor(rows): return '<div class="anchor"><b>In the project</b><ul style="margin:6px 0 0;padding-left:20px">'+"".join(f"<li><code>{f}</code>: {t}</li>" for f,t in rows)+"</ul></div>"
def fig(i,cap): return f'<figure><div id="{i}"></div><figcaption>{cap}</figcaption></figure>'
def eq(t): return f"<div class='eq'>{t}</div>"
def read(t): return f"<div class='read'>{t}</div>"
def table(head,rows,cls="props"):
    return f'<div class="scroll"><table class="{cls}"><tr>'+"".join(f"<th>{h}</th>" for h in head)+"</tr>"+"".join("<tr>"+"".join(f"<td>{c}</td>" for c in r)+"</tr>" for r in rows)+"</table></div>"
def rng(i,label,mn,mx,st,val): return f'<label>{label}<input type="range" id="{i}" min="{mn}" max="{mx}" step="{st}" value="{val}"><output id="{i}-o"></output></label>'
def widget(ctl,inner): return f"<div class='widget'><div class='ctl'>{ctl}</div>{inner}</div>"
def pred(i): return f'<div id="{i}"></div>'
P="f1_racer_rl/src/f1_racer_rl/tasks/f1_racer/"
