import subprocess, sys, os, threading, time, math, random, queue, tempfile, wave

def _pip(pkg):
    subprocess.call([sys.executable,"-m","pip","install","--quiet",
                     "--no-warn-script-location",pkg])

def ensure_deps():
    flag=os.path.join(os.path.dirname(os.path.abspath(__file__)),".zia3")
    tried=os.path.exists(flag); missing=[]
    for mod,pkg in [("cv2","opencv-python==4.9.0.80"),("mediapipe","mediapipe==0.10.14"),
                    ("numpy","numpy"),("pyttsx3","pyttsx3"),("requests","requests")]:
        try: __import__(mod)
        except ImportError: missing.append(pkg)
    try:
        from deepface import DeepFace
    except: missing.append("deepface")
    if not missing:
        if os.path.exists(flag): os.remove(flag)
        return
    if tried: print(f"[WARN] missing: {missing}"); return
    [_pip(p) for p in missing]
    open(flag,"w").close()
    sys.exit(subprocess.call([sys.executable]+sys.argv))

ensure_deps()

import cv2, numpy as np, mediapipe as mp, pyttsx3, requests
os.environ.update({"GLOG_minloglevel":"3","TF_CPP_MIN_LOG_LEVEL":"3",
                   "MEDIAPIPE_DISABLE_GPU":"1"})

API_KEY = os.environ.get("ANTHROPIC_API_KEY","")

W=(255,255,255); K=(0,0,0); G=(60,220,60); C=(255,220,80)
Y=(0,220,220); O=(30,160,255); R=(40,40,220)
PILL_COLS=[G,C,(150,220,255),Y,O,R,(180,100,255),W]
EMO_COLS={"Happy":(60,220,60),"Sad":(220,100,60),"Neutral":(180,180,180),
          "Angry":(40,40,220),"Questioning":(30,160,255),"Skeptical":(0,190,220)}

FACE_DOTS=list(set([
    33,246,161,160,159,158,157,173,133,362,398,384,385,386,387,388,466,263,
    70,63,105,66,107,336,296,334,293,300,168,6,197,195,5,4,1,
    48,64,98,97,2,326,327,294,278,
    61,185,40,39,37,0,267,269,270,409,291,146,91,181,84,17,314,405,321,375,
    78,191,80,81,82,13,312,311,310,415,308,
    172,136,150,149,176,148,152,377,400,365,
    116,123,147,213,345,352,376,433,10,338,297,332,284,109,67,103,54,21,
]))

# Gesture table — your 6 primary signs first
GTABLE = [
    ("TECHNOLOGY",[1,1,1,1,1], 1),  # open hand
    ("USE",       [0,1,1,0,0], 1),  # peace V
    ("FOR",       [0,1,0,0,0], 1),  # point
    ("HELP",      [1,0,0,0,0], 1),  # thumbs up
    ("NOT",       [0,1,0,0,1], 1),  # rock
    ("WAR",       [0,0,0,0,0], 1),  # fist
    # extras
    ("LIFE",      [0,1,1,1,1], 1),
    ("IMPROVE",   [0,0,1,1,0], 1),
    ("WORLD",     [0,0,0,1,1], 1),
    ("LOVE",      [1,1,0,0,1], 1),
    ("GOOD",      [1,1,0,0,0], 1),
    ("LEARN",     [1,0,1,0,0], 1),
    ("CARE",      [1,0,0,1,0], 1),
    ("VOICE",     [1,0,0,0,1], 1),
    ("AI",        [0,1,0,1,0], 1),
    ("HUMAN",     [0,0,1,0,0], 1),
    ("SMART",     [0,0,0,0,1], 1),
    ("THANK",     [0,0,0,1,0], 1),
    ("HELLO",     [1,1,1,1,1], 2),  # both hands open
]

def classify(fingers, num_hands):
    if not fingers or num_hands == 0: return None
    f = [int(b) for b in fingers[0]]
    for word, pat, minh in GTABLE:
        if num_hands >= minh and f == pat:
            return word
    best, bd = None, 999
    for word, pat, minh in GTABLE:
        if minh > 1: continue
        d = sum(a!=b for a,b in zip(f, pat))
        if d < bd: bd, best = d, word
    return best if bd <= 1 else None

F=cv2.FONT_HERSHEY_SIMPLEX
def tsz(t,s): return cv2.getTextSize(t,F,s,1)[0]
def ptxt(img,t,pos,sc,col,th=1,sh=True):
    x,y=int(pos[0]),int(pos[1])
    if sh: cv2.putText(img,t,(x+1,y+1),F,sc,K,th+2,cv2.LINE_AA)
    cv2.putText(img,t,(x,y),F,sc,col,th,cv2.LINE_AA)
def pbold(img,t,pos,sc,col):
    x,y=int(pos[0]),int(pos[1])
    cv2.putText(img,t,(x+2,y+2),F,sc,K,4,cv2.LINE_AA)
    cv2.putText(img,t,(x,y),F,sc,col,2,cv2.LINE_AA)
def prect(img,x1,y1,x2,y2,col,a=0.85,bc=None):
    x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
    x1=max(0,x1);y1=max(0,y1)
    x2=min(img.shape[1]-1,x2);y2=min(img.shape[0]-1,y2)
    if x2<=x1 or y2<=y1: return
    roi=img[y1:y2+1,x1:x2+1]                    # blend only the small box, not the whole frame
    ov=roi.copy(); cv2.rectangle(ov,(0,0),(x2-x1,y2-y1),col,-1)
    cv2.addWeighted(ov,a,roi,1-a,0,roi)
    if bc: cv2.rectangle(img,(x1,y1),(x2,y2),bc,1,cv2.LINE_AA)
def gline(img,p1,p2,col):
    pad=6
    ix1=max(0,int(min(p1[0],p2[0]))-pad); iy1=max(0,int(min(p1[1],p2[1]))-pad)
    ix2=min(img.shape[1]-1,int(max(p1[0],p2[0]))+pad); iy2=min(img.shape[0]-1,int(max(p1[1],p2[1]))+pad)
    if ix2<=ix1 or iy2<=iy1: return
    q1=(int(p1[0])-ix1,int(p1[1])-iy1); q2=(int(p2[0])-ix1,int(p2[1])-iy1)
    roi=img[iy1:iy2+1,ix1:ix2+1]                # blend only the bone's region, not the whole frame
    ov=roi.copy(); cv2.line(ov,q1,q2,col,5,cv2.LINE_AA)
    cv2.addWeighted(ov,0.15,roi,0.85,0,roi)
    cv2.line(img,(int(p1[0]),int(p1[1])),(int(p2[0]),int(p2[1])),col,1,cv2.LINE_AA)
def gdot(img,x,y,r,col):
    h,w=img.shape[:2]; pad=r+5
    ix1=max(0,int(x)-pad); iy1=max(0,int(y)-pad)
    ix2=min(w-1,int(x)+pad); iy2=min(h-1,int(y)+pad)
    if ix2<=ix1 or iy2<=iy1: return
    cx,cy=int(x)-ix1,int(y)-iy1
    roi=img[iy1:iy2+1,ix1:ix2+1]                # blend only the dot's region, not the whole frame
    ov=roi.copy(); cv2.circle(ov,(cx,cy),r+4,col,-1,cv2.LINE_AA)
    cv2.addWeighted(ov,0.15,roi,0.85,0,roi)
    cv2.circle(img,(int(x),int(y)),r,col,-1,cv2.LINE_AA)

BONES=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
       (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
       (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]
TIPS={4,8,12,16,20}

class BgDetector:
    def __init__(self):
        self._q=queue.Queue(maxsize=1); self._eq=queue.Queue(maxsize=1)
        self._fq=queue.Queue(maxsize=1)
        self._lk=threading.Lock(); self._run=True
        self.hand_pts=[]; self.fingers=[]; self.both_open=False
        self.num_hands=0; self.face_dots=[]; self.face_box=None
        self.emotion="Neutral"
        self.emo_sc={"Happy":0,"Sad":0,"Neutral":1,"Angry":0,"Questioning":0,"Skeptical":0}
        self._smooth=[{},{}]
        threading.Thread(target=self._loop,  daemon=True).start()
        threading.Thread(target=self._floop, daemon=True).start()
        threading.Thread(target=self._emloop,daemon=True).start()

    def push(self,small,tiny):
        try: self._q.put_nowait((small,tiny))
        except queue.Full: pass

    def _loop(self):
        hands=mp.solutions.hands.Hands(static_image_mode=False,max_num_hands=2,
            min_detection_confidence=0.6,min_tracking_confidence=0.55)
        A=0.55   # higher = snappier tracking (was 0.35, felt laggy)
        while self._run:
            try: small,tiny=self._q.get(timeout=0.5)
            except queue.Empty: continue
            rgb=cv2.cvtColor(small,cv2.COLOR_BGR2RGB); sh,sw=small.shape[:2]
            hr=hands.process(rgb)
            new_pts=[]; new_fs=[]; both_open=False
            if hr.multi_hand_landmarks:
                for hi,hlm in enumerate(hr.multi_hand_landmarks[:2]):
                    pts={}; sm=self._smooth[hi]
                    for i,lm in enumerate(hlm.landmark):
                        raw=np.array([lm.x*sw,lm.y*sh]); prev=sm.get(i,raw)
                        s=A*raw+(1-A)*prev; sm[i]=s; pts[i]=s
                    self._smooth[hi]=sm; new_pts.append(pts)
                    fs=[]
                    for tip,base in [(4,3),(8,5),(12,9),(16,13),(20,17)]:
                        fs.append(bool(pts[tip][1]<pts[base][1]-5))
                    new_fs.append(fs)
                if len(new_fs)==2 and all(new_fs[0]) and all(new_fs[1]):
                    both_open=True
            try: self._fq.put_nowait(small)   # face mesh runs on its own thread now
            except queue.Full: pass
            if tiny is not None:
                try: self._eq.put_nowait(tiny)
                except queue.Full: pass
            with self._lk:
                self.hand_pts=new_pts; self.fingers=new_fs; self.both_open=both_open
                self.num_hands=len(new_pts)

    def _floop(self):
        face=mp.solutions.face_mesh.FaceMesh(static_image_mode=False,max_num_faces=1,
            refine_landmarks=False,min_detection_confidence=0.5,min_tracking_confidence=0.45)
        while self._run:
            try: small=self._fq.get(timeout=0.5)
            except queue.Empty: continue
            rgb=cv2.cvtColor(small,cv2.COLOR_BGR2RGB); sh,sw=small.shape[:2]
            try: fr=face.process(rgb)
            except Exception: continue
            new_dots=[]; new_box=None
            if fr and fr.multi_face_landmarks:
                flm=fr.multi_face_landmarks[0]
                all_p=[(int(lm.x*sw),int(lm.y*sh)) for lm in flm.landmark]
                for idx in FACE_DOTS:
                    if idx<len(all_p): new_dots.append(all_p[idx])
                xs=[p[0] for p in all_p]; ys=[p[1] for p in all_p]
                new_box=(min(xs)-10,min(ys)-10,max(xs)+10,max(ys)+10)
            if new_dots or new_box:
                with self._lk:
                    if new_dots: self.face_dots=new_dots
                    if new_box:  self.face_box=new_box

    def _emloop(self):
        df=None
        try:
            from deepface import DeepFace; df=DeepFace
        except: pass
        while self._run:
            try: frame=self._eq.get(timeout=1.0)
            except queue.Empty: continue
            if df:
                try:
                    r=df.analyze(frame,actions=['emotion'],enforce_detection=False,silent=True)
                    if r:
                        em=r[0]['dominant_emotion'].lower(); raw=r[0]['emotion']
                        m={'happy':'Happy','sad':'Sad','neutral':'Neutral','angry':'Angry',
                           'fear':'Questioning','disgust':'Skeptical','surprise':'Happy'}
                        tot=sum(raw.values()) or 1
                        with self._lk:
                            self.emotion=m.get(em,'Neutral')
                            self.emo_sc={'Happy':(raw.get('happy',0)+raw.get('surprise',0))/tot,
                                'Sad':raw.get('sad',0)/tot,'Neutral':raw.get('neutral',0)/tot,
                                'Angry':raw.get('angry',0)/tot,'Questioning':raw.get('fear',0)/tot,
                                'Skeptical':raw.get('disgust',0)/tot}
                except: pass
            else:
                with self._lk:
                    sc=self.emo_sc
                    for k in sc: sc[k]=max(0,min(1,sc[k]+random.uniform(-0.01,0.01)))
                    tot=sum(sc.values()) or 1
                    for k in sc: sc[k]/=tot
                    self.emotion=max(sc,key=sc.get)

    def get(self):
        with self._lk:
            return (list(self.hand_pts),list(self.fingers),self.both_open,
                    self.num_hands,list(self.face_dots),self.face_box,
                    self.emotion,dict(self.emo_sc))

    def draw_hands(self,img,hand_pts,sx,sy,fw,fh):
        for pts in hand_pts:
            for a,b in BONES:
                if a in pts and b in pts:
                    p1=(int(np.clip(pts[a][0]*sx,0,fw-1)),int(np.clip(pts[a][1]*sy,0,fh-1)))
                    p2=(int(np.clip(pts[b][0]*sx,0,fw-1)),int(np.clip(pts[b][1]*sy,0,fh-1)))
                    gline(img,p1,p2,W)
            for i,p in pts.items():
                px=int(np.clip(p[0]*sx,0,fw-1)); py=int(np.clip(p[1]*sy,0,fh-1))
                gdot(img,px,py,4 if i in TIPS else 3,W if i in TIPS else (200,200,220))

    def draw_face(self,img,dots,box,emotion,sx,sy,fw,fh):
        for px,py in dots:
            x=int(np.clip(px*sx,0,fw-1)); y=int(np.clip(py*sy,0,fh-1))
            cv2.circle(img,(x,y),1,(200,200,220),-1,cv2.LINE_AA)
        if box:
            x1,y1,x2,y2=box
            x1=int(np.clip(x1*sx,0,fw-1)); y1=int(np.clip(y1*sy,0,fh-1))
            x2=int(np.clip(x2*sx,0,fw-1)); y2=int(np.clip(y2*sy,0,fh-1))
            cv2.rectangle(img,(x1,y1),(x2,y2),W,1,cv2.LINE_AA)
            tw,th=tsz(emotion,0.42); ex=(x1+x2)//2-tw//2; ey=y1-8
            if ey>10:
                prect(img,ex-6,ey-th-4,ex+tw+6,ey+4,K,0.70)
                ptxt(img,emotion,(ex,ey),0.42,W,1)

    def stop(self): self._run=False

class Signs:
    HOLD_TIME = 1.0   # was 1.5s — snappier locking, still resists flicker
    GAP_TIME  = 0.25

    def __init__(self):
        self.words=[]; self._state="IDLE"; self._word=None
        self._timer=0.0; self._gap=0.0; self.prog=0.0

    def update(self, fingers, num_hands, dt):
        word         = classify(fingers, num_hands) if num_hands > 0 else None
        hand_present = (num_hands > 0)

        if self._state == "IDLE":
            self.prog = 0.0
            if word and hand_present:
                self._word=word; self._timer=0.0; self._gap=0.0
                self._state="HOLDING"

        elif self._state == "HOLDING":
            if word == self._word and hand_present:
                self._timer += dt
                self.prog = min(1.0, self._timer/self.HOLD_TIME)
                if self._timer >= self.HOLD_TIME:
                    self._state="LOCKED"; self.prog=1.0
                    w=self._word; self.words.append(w)
                    if len(self.words)>10: self.words=self.words[-10:]
                    print(f"[LOCKED] {w}  →  {self.words}")
                    return w
            else:
                self._word=word; self._timer=0.0; self.prog=0.0
                if not (word and hand_present): self._state="IDLE"

        elif self._state == "LOCKED":
            self.prog = 0.0
            # KEY FIX: physical absence only
            if not hand_present:
                self._gap += dt
                if self._gap >= self.GAP_TIME:
                    self._state="IDLE"; self._gap=0.0; self._word=None
                    print("[READY] — make your next sign")
            else:
                self._gap = 0.0

        return None

    @property
    def current(self): return self._word
    @property
    def state(self):   return self._state

    def reset(self):
        self.words=[]; self._state="IDLE"; self._word=None
        self._timer=0.0; self._gap=0.0; self.prog=0.0

# ── LLM ────────────────────────────────────────────────────────────────────────
class LLM:
    def __init__(self):
        self._s=""; self._pending=False; self._done_words=[]

    def request(self, words):
        if self._pending: return
        if list(words)==self._done_words: return
        self._done_words=list(words); self._pending=True
        threading.Thread(target=self._call,args=(list(words),),daemon=True).start()

    def get_blocking(self, words, timeout=4.0):
        if self._s and self._done_words==list(words): return self._s
        self._done_words=list(words); self._pending=True
        threading.Thread(target=self._call,args=(list(words),),daemon=True).start()
        dl=time.time()+timeout
        while self._pending and time.time()<dl: time.sleep(0.05)
        return self._s or self._fb(words)

    def _call(self, words):
        g=" ".join(words)
        try:
            if API_KEY:
                r=requests.post("https://api.anthropic.com/v1/messages",
                    headers={"x-api-key":API_KEY,"anthropic-version":"2023-06-01",
                             "content-type":"application/json"},
                    json={"model":"claude-haiku-4-5-20251001","max_tokens":60,
                          "messages":[{"role":"user","content":
                              f"Sign language gloss words. Make a natural English sentence "
                              f"(max 12 words): {g}\nReply with ONLY the sentence."}]},
                    timeout=8)
                d=r.json()
                if "content" in d and d["content"]:
                    self._s=d["content"][0]["text"].strip()
                else: self._s=self._fb(words)
            else: self._s=self._fb(words)
        except: self._s=self._fb(words)
        finally: self._pending=False

    def _fb(self, words):
        w=[x.lower() for x in words]
        combos=[
            (["use","technology","for","help","not","war"],
             "Use technology to create help, not war."),
            (["use","technology","help","not","war"],
             "Use technology to create help, not war."),
            (["technology","for","help","not","war"],
             "Use technology to create help, not war."),
            (["use","technology","not","war"],
             "Use technology for good, not war."),
            (["use","technology","help"],   "Use technology to help people."),
            (["technology","not","war"],    "Technology should help, not cause war."),
            (["help","not","war"],          "Help people, not wage war."),
            (["not","war"],                 "Choose peace, not war."),
            (["use","technology"],          "Use technology for good."),
        ]
        for keys, sentence in combos:
            if all(k in w for k in keys): return sentence
        kw=[x for x in words if x.lower() not in {"and","the","a","is"}]
        return (" ".join(kw[:6]).lower().capitalize()+"." if kw else "Sign language to voice.")

    def reset(self): self._s=""; self._pending=False; self._done_words=[]

    @property
    def sentence(self): return self._s
    @property
    def pending(self):  return self._pending

# ── VOICE ──────────────────────────────────────────────────────────────────────
class Voice:
    # KEY FIX: synthesis is written to a WAV file (SAPI file output — no COM
    # audio session involved), then stdlib winsound plays it. If speech is ever
    # inaudible again, [SPEAK-DONE rms=...] proves whether synthesis worked
    # (rms>0) and the problem is Windows output config, not this app.
    SAY_TIMEOUT = 15.0
    WAV = os.path.join(tempfile.gettempdir(), "tg_tts.wav")

    def __init__(self):
        self._q=queue.Queue(); self._on=False; self._last_s=""
        threading.Thread(target=self._run,daemon=True).start()

    def _synth_sapi(self, text, path):
        import pythoncom
        from win32com.client import Dispatch
        pythoncom.CoInitialize()
        try:
            v=Dispatch('SAPI.SpVoice'); v.Rate=1; v.Volume=100
            stream=Dispatch('SAPI.SpFileStream')
            stream.Open(path, 3)   # 3 = SSFMCreateForWrite; bare Open() fails on this SAPI build
            v.AudioOutputStream=stream
            v.Speak(text)                                  # sync write into wav
            stream.Close()
            try: v.AudioOutputStream=None
            except Exception: pass
        finally:
            pythoncom.CoUninitialize()

    def _wav_dur_rms(self, path):
        with wave.open(path,'rb') as w:
            n=w.getnframes(); fr=w.getframerate() or 1
            data=w.readframes(n)
        a=np.frombuffer(data,dtype=np.int16).astype(np.float32)
        rms=float(np.sqrt(np.mean(a*a))) if a.size else 0.0
        return n/float(fr), rms

    def _run(self):
        while True:
            t=self._q.get()
            if t is None: break
            self._on=True
            t0=time.time()
            try:
                self._synth_sapi(t, self.WAV)
                dur,rms=self._wav_dur_rms(self.WAV)
                if rms<50 or dur<0.05:                     # synthesis produced silence
                    print(f"[TTS EMPTY rms={rms:.0f}] trying pyttsx3 fallback")
                    eng=pyttsx3.init(); eng.setProperty('rate',165)
                    eng.say(t); eng.runAndWait()
                    try: eng.stop()
                    except Exception: pass
                else:
                    import winsound
                    winsound.PlaySound(self.WAV,
                        winsound.SND_FILENAME|winsound.SND_ASYNC)
                    time.sleep(min(dur+0.15, self.SAY_TIMEOUT))  # serial queue pacing
                    print(f"[SPEAK-DONE {time.time()-t0:.1f}s rms={rms:.0f}] {t[:48]}")
            except Exception as ex:
                print(f"[TTS FAIL] {ex}")
            self._on=False

    def _drain(self):
        while not self._q.empty():
            try: self._q.get_nowait()
            except: break

    def word(self, w):
        self._drain()
        if w:
            print(f"[SPEAK-WORD] {w}")
            self._last_s = w  # Only track last word for word speaking
            self._q.put(w)

    def sentence(self, s, force=False):
        # Always speak the sentence on finish gesture, do not suppress repeats
        if s:
            self._drain()
            print(f"[SPEAK-FULL] {s}")
            self._q.put(s)

    def reset(self): self._last_s=""

    @property
    def speaking(self): return self._on

# ── UI ─────────────────────────────────────────────────────────────────────────
def ui_top(img, words, sign, state, prog, t):
    h,iw=img.shape[:2]
    prect(img,0,0,iw,38,(15,15,15),0.85)
    ptxt(img,"GLOSS",(8,26),0.38,(140,140,140),1,False)
    if not words and state=="IDLE":
        ptxt(img,"waiting for signs...",(68,26),0.35,(100,100,100),1,False)
    x=68
    for i,w in enumerate(words[-8:]):
        col=PILL_COLS[i%len(PILL_COLS)]; tw,_=tsz(w,0.40)
        cv2.circle(img,(x+4,18),4,col,-1,cv2.LINE_AA)
        ptxt(img,w,(x+12,26),0.40,W,1,False); x+=tw+24
    prect(img,iw-52,6,iw-4,32,(40,40,40),0.80,bc=(100,100,100))
    ptxt(img,"Clear",(iw-48,24),0.34,(180,180,180),1,False)
    ti="Zia's Glasses"; tw2,_=tsz(ti,0.38)
    ptxt(img,ti,(iw//2-tw2//2,24),0.38,(180,180,180),1,False)
    ts=time.strftime("%a %H:%M"); tw3,_=tsz(ts,0.34)
    ptxt(img,ts,(iw-tw3-60,24),0.34,(160,160,160),1,False)

def ui_sentence(img, sentence, pending, t):
    """Sentence — positioned below top bar, fully visible, never behind bottom."""
    if not sentence: return
    h,iw=img.shape[:2]
    words=sentence.split(); lines=[]; line=""
    for ww in words:
        test=(line+" "+ww).strip()
        if tsz(test,0.72)[0]>iw-80 and line: lines.append(line); line=ww
        else: line=test
    if line: lines.append(line)
    y=55  # starts just below 38px top bar
    for ln in lines[:3]:
        tw,th=tsz(ln,0.72)
        prect(img,0,y-th-4,tw+20,y+6,K,0.60)   # ROI blend (was full-frame copy per line)
        cv2.putText(img,ln,(11,y+1),F,0.72,K,3,cv2.LINE_AA)
        cv2.putText(img,ln,(10,y),  F,0.72,W,2,cv2.LINE_AA)
        y+=th+10
    if pending:
        cv2.circle(img,(20,y+10),5,C,-1,cv2.LINE_AA)
        ptxt(img,"interpreting...",(32,y+15),0.35,C,1,False)

def ui_lock(img, word, state, prog, t):
    if not word: return
    h,iw=img.shape[:2]
    tw,_=tsz(word,0.80); bw=max(tw+40,280); bh=70
    bx=iw//2-bw//2
    by=h-bh-115  # above bottom bar (100px) + emotion panel gap
    col_border=(60,220,60) if state=="LOCKED" else (70,70,70)
    prect(img,bx,by,bx+bw,by+bh,K,0.90,bc=col_border)
    pbold(img,word,(bx+15,by+38),0.80,W)
    if state=="HOLDING":
        bar=bw-30; filled=int(bar*prog)
        cv2.rectangle(img,(bx+15,by+48),(bx+15+bar,by+56),(40,40,40),-1)
        if filled>0: cv2.rectangle(img,(bx+15,by+48),(bx+15+filled,by+56),(60,180,60),-1)
        ptxt(img,f"{int(prog*100)}%  hold still...",(bx+15,by+66),0.30,(120,120,120),1,False)
    elif state=="LOCKED":
        ptxt(img,"LOCKED! lower hand for next sign",(bx+15,by+66),0.30,(60,220,60),1,False)

def ui_finish(img, prog, t):
    h,iw=img.shape[:2]; bw=360; bh=70
    bx=iw//2-bw//2; by=h-bh-115
    prect(img,bx,by,bx+bw,by+bh,K,0.92,bc=(60,220,60))
    pbold(img,"FINISH  -  hold both hands open",(bx+15,by+36),0.55,(60,220,60))
    bar=bw-30; filled=int(bar*prog)
    cv2.rectangle(img,(bx+15,by+46),(bx+15+bar,by+54),(40,40,40),-1)
    if filled>0: cv2.rectangle(img,(bx+15,by+46),(bx+15+filled,by+54),(60,220,60),-1)
    ptxt(img,f"{int(prog*100)}%",(bx+15,by+66),0.32,(120,200,120),1,False)

def ui_emotion(img, scores, t):
    """RIGHT side overlay, positioned so it doesn't overlap bottom bar."""
    h,iw=img.shape[:2]
    pw=155; ph=165
    px=iw-pw-10
    # sits just above bottom bar
    py=h-100-ph-12
    prect(img,px-4,py-8,px+pw+4,py+ph,K,0.82,bc=(50,50,50))
    order=["Happy","Sad","Neutral","Angry","Questioning","Skeptical"]
    top=max(scores,key=scores.get) if scores else "Neutral"
    ey=py
    for em in order:
        sc=scores.get(em,0); col=EMO_COLS.get(em,W); is_top=(em==top)
        cv2.circle(img,(px+8,ey+10),5,col,-1,cv2.LINE_AA)
        if is_top: cv2.circle(img,(px+8,ey+10),7,col,1,cv2.LINE_AA)
        ptxt(img,em,(px+18,ey+15),0.36,W if is_top else (150,150,150),1,False)
        bw2=int(sc*75)
        if bw2>0: cv2.rectangle(img,(px+90,ey+6),(px+90+bw2,ey+14),col,-1)
        ey+=26

def ui_bottom(img, mode, speaking, t):
    h,iw=img.shape[:2]; bh=100; by=h-bh
    prect(img,0,by,iw,h,(5,5,5),0.92)
    labels=["Hand\nTracking","Emotion\nDetection","LLM\nInterpretation","Voice\nOutput"]
    active={"Hand\nTracking":True,"Emotion\nDetection":True,
            "LLM\nInterpretation":mode>=2,"Voice\nOutput":speaking}
    mw=iw//4
    for i,m in enumerate(labels):
        mx=i*mw; on=active.get(m,False)
        if on: prect(img,mx+6,by+6,mx+mw-6,h-6,(40,40,40),0.90,bc=(80,80,80))
        col=W if on else (80,80,80)
        for li,line in enumerate(m.split("\n")):
            lw,_=tsz(line,0.44); lx=mx+mw//2-lw//2; ly=by+32+li*20
            ptxt(img,line,(lx,ly),0.44,col,1,False)
        if on and m=="Voice\nOutput":
            dcx=mx+mw//2; dcy=h-22
            for di in range(11):
                dx=dcx-25+di*5; amp=abs(math.sin(t*4+di*0.5))*8+2
                cv2.circle(img,(dx,dcy),int(amp/3)+1,W,-1,cv2.LINE_AA)
        elif on and m=="Emotion\nDetection":
            sq=12; sx2=mx+mw//2-sq//2; sy2=h-22-sq//2
            cv2.rectangle(img,(sx2,sy2),(sx2+sq,sy2+sq),W,-1)

# ── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    print("\n╔══════════════════════════════════════╗")
    print("║        Zia's Glasses                ║")
    print("╚══════════════════════════════════════╝")
    print()
    print("  YOUR 6 SIGNS:")
    print("  peace-V       → USE")
    print("  open hand     → TECHNOLOGY")
    print("  point index   → FOR")
    print("  thumbs up     → HELP")
    print("  index+pinky   → NOT")
    print("  fist          → WAR")
    print()
    print("  Hold 1.0s → LOCKED. Drop hand → next sign.")
    print("  Both hands open 1.5s → FINISH → speaks sentence")
    print("  C=clear  Q/ESC=quit\n")

    cap=cv2.VideoCapture(0)
    if not cap.isOpened(): print("no webcam"); sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)   # 60 rarely delivered by webcams; asking for it adds latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cv2.namedWindow("Zia's Glasses", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Zia's Glasses", 1280, 720)

    det=BgDetector(); signs=Signs(); llm=LLM(); voice=Voice()
    prev=time.time(); fps_s=30.0; t=0.0
    sentence=""; mode=0; fin_timer=0.0; fin_done=False; FIN_THRESH=1.5   # was 2.0s — quicker sentence finish
    PW,PH=480,270   # mediapipe downscales to ~256px internally; smaller = faster handoff

    _clr=[False]
    def mouse_cb(event,mx,my,flags,param):
        if event==cv2.EVENT_LBUTTONDOWN:
            iw2=param[0]
            if iw2-52<=mx<=iw2-4 and 6<=my<=32: _clr[0]=True
    cv2.setMouseCallback("Zia's Glasses",mouse_cb,[1280])

    while True:
        ret,frame=cap.read()
        if not ret: break
        frame=cv2.flip(frame,1); FH,FW=frame.shape[:2]
        now=time.time(); dt=min(now-prev,0.1); prev=now; t+=dt
        fps_s=fps_s*0.9+(1/dt)*0.1

        small=cv2.resize(frame,(PW,PH)); tiny=cv2.resize(frame,(320,180))
        det.push(small,tiny)
        hand_pts,fingers,both_open,num_hands,face_dots,face_box,emotion,emo_sc=det.get()
        sx=FW/PW; sy=FH/PH

        if _clr[0]:
            _clr[0]=False; signs.reset(); llm.reset(); voice.reset()
            sentence=""; mode=0; fin_timer=0.0; fin_done=False; print("[CLEAR]")

        # FINISH gesture
        fp=0.0
        if both_open and signs.words:
            fin_timer = min(fin_timer + dt, FIN_THRESH + 0.1); fp = min(1.0, fin_timer / FIN_THRESH)
            if fp >= 1.0 and not fin_done:
                print("[FINISH GESTURE DETECTED] Speaking full sentence...")
                fin_done = True; mode = 3
                final = llm.get_blocking(signs.words, timeout=4.0)
                sentence = final
                print(f"[SPEAK-FULL-DEBUG] {sentence}")
                voice.sentence(sentence, force=True)  # Always speak full sentence on finish
        else:
            fin_timer=max(0.0,fin_timer-dt*2); fp=fin_timer/FIN_THRESH
            if not both_open: fin_done=False

        # word locking
        show_finish=both_open and bool(signs.words)
        if not both_open:
            locked=signs.update(fingers,num_hands,dt)
            if locked:
                mode=max(mode,1); voice.word(locked)
                llm.request(signs.words); mode=max(mode,2)

        bg=llm.sentence
        if bg and bg!=sentence and not fin_done:
            sentence=bg; mode=max(mode,2)

        # draw
        dark=cv2.convertScaleAbs(frame,alpha=0.60,beta=0)
        det.draw_face(dark,face_dots,face_box,emotion,sx,sy,FW,FH)
        det.draw_hands(dark,hand_pts,sx,sy,FW,FH)

        if show_finish or fin_timer>0.05: ui_finish(dark,fp,t)
        else: ui_lock(dark,signs.current,signs.state,signs.prog,t)

        ui_emotion(dark,emo_sc,t)
        ui_sentence(dark,sentence,llm.pending,t)
        ui_top(dark,signs.words,signs.current,signs.state,signs.prog,t)
        ui_bottom(dark,mode,voice.speaking,t)
        ptxt(dark,f"FPS {fps_s:.0f}",(FW-72,FH-110),0.36,(100,100,100),1,False)

        cv2.imshow("Zia's Glasses",dark)
        key=cv2.waitKey(1)&0xFF
        if key in (ord('q'),ord('Q'),27): break
        elif key in (ord('c'),ord('C')):
            signs.reset(); llm.reset(); voice.reset()
            sentence=""; mode=0; fin_timer=0.0; fin_done=False; print("[CLEAR]")

    det.stop(); cap.release(); cv2.destroyAllWindows(); print("Bye!")

if __name__=="__main__":
    main()