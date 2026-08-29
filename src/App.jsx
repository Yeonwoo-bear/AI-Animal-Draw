import { useEffect, useRef, useState } from "react";
import { FilesetResolver, HandLandmarker, DrawingUtils } from "@mediapipe/tasks-vision";
import { ANIMALS } from "./animals";
import { loadAnimalModel, predictAnimals } from "./classifier";
import { getScores, saveScore } from "./leaderboard";

const LIMIT = 60;
const MIN_PREDICT_POINTS = 18;

export default function App() {
  const video = useRef();
  const overlay = useRef();
  const paper = useRef();
  const cursor = useRef();
  const stream = useRef();
  const landmarker = useRef();
  const raf = useRef();

  const active = useRef(false);
  const drawOn = useRef(false);
  const ready = useRef(false);
  const missionRef = useRef(null);
  const prev = useRef(null);
  const smooth = useRef(null);
  const stroke = useRef(null);
  const strokes = useRef([]);
  const busy = useRef(false);
  const lastAI = useRef(0);
  const history = useRef([]);
  const started = useRef(0);
  const submitting = useRef(false);

  const [page, setPage] = useState("home");
  const [department, setDepartment] = useState("");
  const [name, setName] = useState("");
  const [phone4, setPhone4] = useState("");
  const [cam, setCam] = useState(false);
  const [hand, setHand] = useState(false);
  const [model, setModel] = useState(false);
  const [drawing, setDrawing] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [mission, setMission] = useState(null);
  const [time, setTime] = useState(LIMIT);
  const [top, setTop] = useState([]);
  const [status, setStatus] = useState("카메라를 연결해주세요.");
  const [board, setBoard] = useState(() => getScores());
  const [result, setResult] = useState(null);

  function clearCanvas() {
    if (paper.current) {
      paper.current.width = 640;
      paper.current.height = 640;
      const ctx = paper.current.getContext("2d");
      ctx.fillStyle = "#fffdfa";
      ctx.fillRect(0, 0, 640, 640);
    }

    if (cursor.current) {
      cursor.current.width = 640;
      cursor.current.height = 640;
      cursor.current.getContext("2d").clearRect(0, 0, 640, 640);
    }

    strokes.current = [];
    stroke.current = null;
    prev.current = null;
    smooth.current = null;
    history.current = [];
    setTop([]);
  }

  function eraseDrawing() {
    clearCanvas();
    if (active.current) setStatus("그림을 지웠어요. SPACE를 눌러 다시 그려보세요.");
  }

  async function connect() {
    try {
      setStatus("카메라 연결 중...");

      stream.current = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
        audio: false,
      });

      video.current.srcObject = stream.current;
      await video.current.play();

      overlay.current.width = video.current.videoWidth;
      overlay.current.height = video.current.videoHeight;
      clearCanvas();

      setStatus("손 인식 준비 중...");

      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm"
      );

      landmarker.current = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
          delegate: "GPU",
        },
        runningMode: "VIDEO",
        numHands: 1,
        minHandDetectionConfidence: 0.5,
        minHandPresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      setCam(true);
      loop();

      setStatus("8-class 동물 CNN 준비 중...");

      try {
        await loadAnimalModel();
        ready.current = true;
        setModel(true);
        setStatus("준비 완료! 게임 시작을 눌러주세요.");
      } catch (modelError) {
        console.error("LITERT MODEL ERROR:", modelError);
        ready.current = false;
        setModel(false);
        setStatus("손 인식은 정상입니다. animal_draw_v11.tflite 파일을 확인해주세요.");
      }
    } catch (error) {
      console.error(error);
      setStatus(`오류: ${error?.message || error}`);
    }
  }

  const filtered = (a, b) =>
    !a ? b : { x: a.x * 0.18 + b.x * 0.82, y: a.y * 0.18 + b.y * 0.82 };

  function drawCursor(p) {
    const canvas = cursor.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, 640, 640);

    ctx.beginPath();
    ctx.arc(p.x, p.y, 18, 0, Math.PI * 2);
    ctx.strokeStyle = drawOn.current ? "#6d9278" : "#b2a69c";
    ctx.lineWidth = 3;
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(p.x, p.y, 6, 0, Math.PI * 2);
    ctx.fillStyle = drawOn.current ? "#6d9278" : "#776c63";
    ctx.fill();
  }

  function segment(a, b) {
    const ctx = paper.current.getContext("2d");
    ctx.strokeStyle = "#342e2a";
    ctx.lineWidth = 8;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }

  function stabilize(arr) {
    history.current.push(arr);
    if (history.current.length > 3) history.current.shift();

    const map = new Map();

    for (const frame of history.current) {
      for (const p of frame) map.set(p.id, (map.get(p.id) || 0) + p.p);
    }

    return ANIMALS
      .map((animal) => ({
        ...animal,
        p: (map.get(animal.id) || 0) / history.current.length,
      }))
      .sort((a, b) => b.p - a.p);
  }

  async function infer() {
    const pointCount = strokes.current.flat().length;
    if (!active.current || !ready.current || busy.current || pointCount < MIN_PREDICT_POINTS) return;

    busy.current = true;

    try {
      const raw = await predictAnimals(strokes.current);
      const stable = stabilize(raw);
      setTop(stable.slice(0, 3));

      if (stable[0]) {
        setStatus(`🤖 AI는 지금 ${stable[0].label} ${(stable[0].p * 100).toFixed(0)}%로 보고 있어요.`);
      }
    } catch (error) {
      console.error("PREDICTION ERROR:", error);
    } finally {
      busy.current = false;
    }
  }

  function loop() {
    if (video.current?.readyState >= 2 && landmarker.current) {
      const result = landmarker.current.detectForVideo(video.current, performance.now());
      const ctx = overlay.current.getContext("2d");
      ctx.clearRect(0, 0, overlay.current.width, overlay.current.height);

      if (result.landmarks?.length) {
        setHand(true);

        const lm = result.landmarks[0];
        const drawingUtils = new DrawingUtils(ctx);
        drawingUtils.drawConnectors(lm, HandLandmarker.HAND_CONNECTIONS, { lineWidth: 3 });
        drawingUtils.drawLandmarks(lm, { radius: 4 });

        const raw = { x: (1 - lm[8].x) * 640, y: lm[8].y * 640 };
        const p = filtered(smooth.current, raw);
        smooth.current = p;
        drawCursor(p);

        if (active.current && drawOn.current) {
          if (!stroke.current) {
            stroke.current = [];
            strokes.current.push(stroke.current);
          }

          stroke.current.push({ ...p });
          if (prev.current) segment(prev.current, p);
          prev.current = p;
        } else {
          prev.current = null;
        }

        const now = performance.now();
        if (
          active.current &&
          strokes.current.flat().length >= MIN_PREDICT_POINTS &&
          now - lastAI.current > 900
        ) {
          lastAI.current = now;
          infer();
        }
      } else {
        setHand(false);
        prev.current = null;
      }
    }

    raf.current = requestAnimationFrame(loop);
  }

  function start() {
    if (!cam) {
      setStatus("먼저 카메라를 연결해주세요.");
      return;
    }

    if (!model) {
      setStatus("동물 AI 모델이 아직 준비되지 않았어요.");
      return;
    }

    clearCanvas();

    const next = ANIMALS[Math.floor(Math.random() * ANIMALS.length)];
    missionRef.current = next;
    setMission(next);
    setTime(LIMIT);
    setResult(null);
    started.current = performance.now();
    submitting.current = false;
    active.current = true;
    drawOn.current = false;
    setPlaying(true);
    setDrawing(false);
    setPage("game");
    setStatus(`${next.label}을(를) 그려주세요. 최대 60초! SPACE로 그리기를 시작하세요.`);
  }

  function toggle() {
    if (!active.current) return;

    const next = !drawOn.current;
    drawOn.current = next;
    setDrawing(next);
    prev.current = null;

    if (next) {
      stroke.current = [];
      strokes.current.push(stroke.current);
      setStatus("DRAW ON · 검지로 그려보세요.");
    } else {
      stroke.current = null;
      setStatus("DRAW OFF · 다시 SPACE를 누르면 이어서 그릴 수 있어요.");
      infer();
    }
  }

  async function submitDrawing(forceElapsed = null) {
    if (submitting.current || !missionRef.current) return;

    submitting.current = true;
    active.current = false;
    drawOn.current = false;
    setPlaying(false);
    setDrawing(false);
    stroke.current = null;

    const elapsed = forceElapsed ?? Math.min(LIMIT, (performance.now() - started.current) / 1000);
    const pointCount = strokes.current.flat().length;

    if (pointCount < 2) {
      setResult({ empty: true, elapsed });
      setPage("result");
      submitting.current = false;
      return;
    }

    try {
      setStatus("최종 그림을 AI가 분석하고 있어요...");
      const predictions = await predictAnimals(strokes.current);
      const target = missionRef.current;
      const targetPrediction = predictions.find((p) => p.id === target.id);
      const similarity = Math.max(0, Math.min(1, targetPrediction?.p ?? 0));
      const finalTop = predictions.slice(0, 3);

      const saved = saveScore({
        department: department.trim(),
        name: name.trim(),
        phone4,
        animal: target.label,
        animalId: target.id,
        similarity,
        elapsed,
        top3: finalTop.map((p) => ({ id: p.id, label: p.label, p: p.p })),
      });

      setBoard(saved.rows);
      setResult({
        ...saved.row,
        rank: saved.rank,
        top3: finalTop,
      });
      setTop(finalTop);
      setPage("result");
    } catch (error) {
      console.error("FINAL PREDICTION ERROR:", error);
      setStatus(`최종 분석 오류: ${error?.message || error}`);
      active.current = true;
      setPlaying(true);
    } finally {
      submitting.current = false;
    }
  }

  useEffect(() => {
    if (!playing) return;

    const id = setInterval(() => {
      setTime((current) => {
        if (current <= 1) {
          setTimeout(() => submitDrawing(LIMIT), 0);
          return 0;
        }
        return current - 1;
      });
    }, 1000);

    return () => clearInterval(id);
  }, [playing]);

  useEffect(() => {
    const onKeyDown = (event) => {
      if (event.code !== "Space" || event.repeat) return;
      if (["INPUT", "TEXTAREA", "BUTTON"].includes(event.target.tagName)) return;
      event.preventDefault();
      toggle();
    };

    addEventListener("keydown", onKeyDown);
    return () => removeEventListener("keydown", onKeyDown);
  }, []);

  useEffect(
    () => () => {
      cancelAnimationFrame(raf.current);
      stream.current?.getTracks?.().forEach((track) => track.stop());
      landmarker.current?.close?.();
    },
    []
  );

  const formValid =
    department.trim().length >= 2 &&
    name.trim().length >= 2 &&
    /^\d{4}$/.test(phone4);

  if (page === "home") {
    return (
      <main className="center">
        <section className="home">
          <span className="eyebrow">AI CONVERGENCE · COMPUTER VISION</span>
          <h1>AI ANIMAL <i>DRAW</i></h1>
          <p>
            제시된 동물을 허공에 그려보세요. <b>AI가 얼마나 비슷하게 인식하는지</b> 점수로 경쟁합니다.
          </p>

          <div className="animals">
            {ANIMALS.map((animal) => (
              <span key={animal.id}>
                {animal.emoji}
                <small>{animal.label}</small>
              </span>
            ))}
          </div>

          <div className="formGrid">
            <label>
              <span>학과</span>
              <input
                value={department}
                onChange={(e) => setDepartment(e.target.value.slice(0, 24))}
                placeholder="예: AI융합학부"
              />
            </label>

            <label>
              <span>닉네임</span>
              <input
                value={name}
                onChange={(e) => setName(e.target.value.slice(0, 12))}
                placeholder="2~12자"
              />
            </label>

            <label>
              <span>전화번호 뒤 4자리</span>
              <input
                value={phone4}
                inputMode="numeric"
                onChange={(e) => setPhone4(e.target.value.replace(/\D/g, "").slice(0, 4))}
                placeholder="1234"
              />
            </label>
          </div>

          <p className="privacy">※ 전화번호 전체가 아닌 뒤 4자리만 참가자 확인용으로 사용합니다.</p>

          <button className="primary" disabled={!formValid} onClick={() => setPage("game")}>체험 시작 →</button>
          <button className="secondary" onClick={() => setPage("rank")}>🏆 랭킹 보기</button>
        </section>
      </main>
    );
  }

  if (page === "rank") {
    return (
      <main className="center">
        <section className="rank">
          <header>
            <div>
              <span className="eyebrow">LEADERBOARD</span>
              <h2>🏆 AI ANIMAL DRAW 랭킹</h2>
              <p>AI 일치도가 높을수록 상위 · 동점이면 더 빨리 제출한 참가자가 우선입니다.</p>
            </div>
            <button onClick={() => setPage("home")}>← 처음으로</button>
          </header>

          <div className="rhead">
            <span>순위</span><span>닉네임</span><span>학과</span><span>미션</span><span>AI 일치도</span><span>시간</span>
          </div>

          {board.length ? board.map((row, i) => (
            <div className="rrow" key={row.id}>
              <span>{i < 3 ? ["🥇", "🥈", "🥉"][i] : i + 1}</span>
              <b>{row.name}</b>
              <span>{row.department}</span>
              <span>{ANIMALS.find((a) => a.id === row.animalId)?.emoji} {row.animal}</span>
              <strong>{Math.round((row.similarity || 0) * 100)}%</strong>
              <span>{Number(row.elapsed || 0).toFixed(1)}s</span>
            </div>
          )) : <p className="empty rankEmpty">아직 기록이 없어요.</p>}
        </section>
      </main>
    );
  }

  if (page === "result") {
    if (result?.empty) {
      return (
        <main className="center">
          <section className="result">
            <div className="big">✏️</div>
            <span className="eyebrow">NO DRAWING</span>
            <h2>그림이 아직 없어요!</h2>
            <p>SPACE를 눌러 검지로 그림을 그린 뒤 제출해 주세요.</p>
            <div className="buttons">
              <button onClick={start}>다시 도전</button>
              <button onClick={() => setPage("home")}>처음으로</button>
            </div>
          </section>
        </main>
      );
    }

    const targetAnimal = ANIMALS.find((a) => a.id === result?.animalId);

    return (
      <main className="center">
        <section className="result">
          <div className="big">{targetAnimal?.emoji}</div>
          <span className="eyebrow">AI SIMILARITY RESULT</span>
          <h2>{result?.animal} AI 일치도</h2>
          <div className="similarityNumber">{Math.round((result?.similarity || 0) * 100)}<small>%</small></div>
          <p className="resultCopy">AI가 최종 그림을 <b>{result?.animal}</b>로 인식한 확률이에요.</p>

          <div className="finalPredictions">
            <h3>AI는 이렇게 봤어요</h3>
            {(result?.top3 || []).map((p, i) => (
              <div className="finalPredRow" key={p.id}>
                <span>{i + 1}</span>
                <b>{p.emoji} {p.label}</b>
                <strong>{Math.round(p.p * 100)}%</strong>
              </div>
            ))}
          </div>

          <div className="stats resultStats">
            <div><small>MISSION</small><b>{targetAnimal?.emoji} {result?.animal}</b></div>
            <div><small>TIME</small><b>{Number(result?.elapsed || 0).toFixed(1)}s</b></div>
            <div><small>SIMILARITY</small><b>{Math.round((result?.similarity || 0) * 100)}%</b></div>
            <div><small>RANK</small><b>#{result?.rank}</b></div>
          </div>

          <div className="buttons">
            <button onClick={start}>다시 도전</button>
            <button onClick={() => setPage("rank")}>🏆 랭킹</button>
            <button onClick={() => setPage("home")}>처음으로</button>
          </div>
        </section>
      </main>
    );
  }

  return (
    <main className="game">
      <header className="topbar">
        <div>
          <span className="eyebrow">AI CONVERGENCE · COMPUTER VISION</span>
          <h1>AI ANIMAL <i>DRAW</i></h1>
        </div>

        <div className="playerBadge">
          <small>PLAYER</small>
          <b>{name || "-"}</b>
          <span>{department || "학과 미입력"}</span>
        </div>

        <div className="mission">
          {mission ? <>
            <span>{mission.emoji}</span>
            <div><small>MISSION</small><b>{mission.label} 그리기</b></div>
          </> : <b>READY?</b>}
        </div>

        <div className={`mode ${drawing ? "on" : ""}`}>
          {drawing ? "✏️ DRAW ON" : "◎ DRAW OFF"} <kbd>SPACE</kbd>
        </div>

        <div className={`timer ${time <= 10 ? "danger" : ""}`}>
          <b>{time}</b><small>SEC</small>
        </div>
      </header>

      <section className="layout">
        <article>
          <div className="title">
            <div><b>📷 LIVE CAMERA</b><small>MediaPipe Hand Tracking</small></div>
            <span className={hand ? "ok" : ""}>{hand ? "● HAND DETECTED" : "○ WAITING"}</span>
          </div>
          <div className="camera">
            <video ref={video} muted playsInline />
            <canvas ref={overlay} />
          </div>
        </article>

        <article>
          <div className="title">
            <div><b>✎ YOUR DRAWING</b><small>INDEX FINGER</small></div>
            <span>{drawing ? "DRAWING" : "CURSOR MODE"}</span>
          </div>
          <div className={`canvas ${drawing ? "on" : ""}`}>
            <canvas ref={paper} />
            <canvas ref={cursor} />
          </div>
        </article>

        <aside>
          <section className="panel">
            <span className="eyebrow">CONTROL</span>
            <h2>60초 AI 유사도 챌린지</h2>
            <div className={`space ${drawing ? "on" : ""}`}>SPACE</div>
            <p>{status}</p>

            <div className="controls">
              <button onClick={connect} disabled={cam}>📷 카메라</button>
              <button className="dark" onClick={start}>🎮 게임 시작</button>
              <button onClick={eraseDrawing}>↺ 지우기</button>
              <button onClick={() => setPage("rank")}>🏆 순위</button>
            </div>

            <button className="submitButton" disabled={!playing || strokes.current.flat().length < 2} onClick={() => submitDrawing()}>
              ✓ 그림 제출
            </button>
            <small className="submitHelp">다 그렸다면 60초를 기다리지 않고 제출할 수 있어요.</small>
          </section>

          <section className="panel prediction">
            <div className="phead">
              <div><span className="eyebrow">ANIMAL CNN</span><h2>🤖 실시간 Top 3</h2></div>
              <span className={`live ${model ? "on" : ""}`}>{model ? "LIVE" : "LOADING"}</span>
            </div>

            {top.length ? (
              <div className="preds">
                {top.map((p, i) => (
                  <div className="prow" key={p.id}>
                    <span>{i + 1}</span>
                    <div>
                      <header><b>{p.emoji} {p.label}</b><small>{Math.round(p.p * 100)}%</small></header>
                      <i><em style={{ width: `${Math.max(3, p.p * 100)}%` }} /></i>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="empty">그림을 시작하면<br />AI가 실시간으로 추측해요.</div>
            )}

            <footer>HAND → STROKES → 64×64 → 8-CLASS CNN → SIMILARITY</footer>
          </section>
        </aside>
      </section>
    </main>
  );
}
