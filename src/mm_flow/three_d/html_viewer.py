from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from mm_flow.three_d.collision import CuboidObstacle, CylinderObstacle, Obstacle3D, SphereObstacle, state_clearance
from mm_flow.three_d.kinematics import SimpleMobileManipulator3D


def export_reach_sequence_html(
    robot: SimpleMobileManipulator3D,
    trajectory: np.ndarray,
    goals_xyz: np.ndarray,
    segment_ends: list[int],
    obstacles: list[Obstacle3D],
    output_path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    robot_visuals = robot.export_visual_assets(output_path.parent) if hasattr(robot, "export_visual_assets") else []
    collision_proxies = robot.export_collision_proxies() if hasattr(robot, "export_collision_proxies") else []

    frames = []
    for state in trajectory:
        fk = robot.forward_kinematics(state)
        link_transforms = fk.get("link_transforms", {})
        frames.append(
            {
                "base": fk["base"].tolist(),
                "mount": fk["mount"].tolist(),
                "elbow": fk["elbow"].tolist(),
                "ee": fk["ee"].tolist(),
                "points": fk.get("points", np.array([fk["mount"], fk["elbow"], fk["ee"]])).tolist(),
                "linkTransforms": {name: transform.tolist() for name, transform in link_transforms.items()},
                "frameOrigins": {name: transform[:3, 3].tolist() for name, transform in link_transforms.items()},
                "yaw": float(state[2]),
                "state": np.asarray(state, dtype=float).tolist(),
                "clearance": float(state_clearance(robot, state, obstacles)),
            }
        )

    obstacle_data = []
    for obs in obstacles:
        if isinstance(obs, SphereObstacle):
            obstacle_data.append({"type": "sphere", "center": list(obs.center), "radius": obs.radius})
        elif isinstance(obs, CuboidObstacle):
            obstacle_data.append({"type": "cuboid", "center": list(obs.center), "size": list(obs.size)})
        elif isinstance(obs, CylinderObstacle):
            obstacle_data.append({
                "type": "cylinder",
                "center": list(obs.center),
                "radius": obs.radius,
                "height": obs.height,
            })

    payload = {
        "frames": frames,
        "goals": goals_xyz.tolist(),
        "segmentEnds": segment_ends,
        "obstacles": obstacle_data,
        "baseRadius": robot.base_radius,
        "baseHeight": robot.base_height,
        "metadata": _to_jsonable(metadata or {}),
        "robotVisuals": robot_visuals,
        "collisionProxies": collision_proxies,
    }
    output_path.write_text(_html_template(json.dumps(payload)), encoding="utf-8")


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _html_template(payload_json: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MM-Flow 3D Reach Sequence Viewer</title>
  <style>
    html, body {{
      margin: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f6f6f4;
      color: #1f2933;
    }}
    #viewer {{
      position: fixed;
      inset: 0;
    }}
    #hud {{
      position: fixed;
      left: 16px;
      bottom: 16px;
      right: 16px;
      display: grid;
      grid-template-columns: auto auto auto 1fr auto;
      gap: 10px;
      align-items: center;
      padding: 10px 12px;
      background: rgba(255, 255, 255, 0.88);
      border: 1px solid rgba(31, 41, 51, 0.12);
      border-radius: 8px;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.10);
    }}
    #top {{
      position: fixed;
      top: 14px;
      left: 16px;
      padding: 7px 9px;
      background: rgba(255, 255, 255, 0.88);
      border: 1px solid rgba(31, 41, 51, 0.12);
      border-radius: 8px;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.10);
      font-size: 13px;
      line-height: 1.35;
    }}
    #infoPanel {{
      position: fixed;
      top: 14px;
      right: 16px;
      width: min(290px, calc(100vw - 32px));
      padding: 9px 10px;
      background: rgba(255, 255, 255, 0.86);
      border: 1px solid rgba(31, 41, 51, 0.12);
      border-radius: 8px;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.10);
      font-size: 12px;
      line-height: 1.32;
      pointer-events: none;
    }}
    #infoPanel .title {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 5px;
      font-weight: 700;
    }}
    #infoPanel .grid {{
      display: grid;
      grid-template-columns: auto 1fr;
      column-gap: 9px;
      row-gap: 2px;
      font-variant-numeric: tabular-nums;
    }}
    #infoPanel .key {{
      color: #52606d;
      white-space: nowrap;
    }}
    #infoPanel .value {{
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      text-align: right;
    }}
    button {{
      border: 1px solid #9aa5b1;
      background: #ffffff;
      border-radius: 6px;
      padding: 6px 10px;
      cursor: pointer;
      font-size: 13px;
    }}
    select {{
      border: 1px solid #9aa5b1;
      background: #ffffff;
      border-radius: 6px;
      padding: 6px 8px;
      font-size: 13px;
    }}
    .toggle {{
      display: flex;
      align-items: center;
      gap: 5px;
      white-space: nowrap;
      font-size: 13px;
      color: #1f2933;
      user-select: none;
    }}
    .toggle input {{
      width: 14px;
      height: 14px;
      margin: 0;
    }}
    input[type="range"] {{
      width: 100%;
    }}
    #stepLabel {{
      min-width: 130px;
      text-align: right;
      font-size: 13px;
      font-variant-numeric: tabular-nums;
    }}
  </style>
</head>
<body>
  <div id="viewer"></div>
  <div id="top">
    <strong>MM-Flow 3D Reach</strong>
  </div>
  <div id="infoPanel"></div>
  <div id="hud">
    <button id="play">Play</button>
    <select id="speed" title="Playback speed">
      <option value="0.5">0.5x</option>
      <option value="0.75">0.75x</option>
      <option value="1" selected>1x</option>
      <option value="1.5">1.5x</option>
      <option value="2">2x</option>
    </select>
    <label class="toggle" title="Show collision proxy boxes">
      <input id="showProxies" type="checkbox" checked />
      <span>proxy</span>
    </label>
    <input id="slider" type="range" min="0" max="0" value="0" step="1" />
    <div id="stepLabel">step 0</div>
  </div>

  <script type="importmap">
    {{
      "imports": {{
        "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
        "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
      }}
    }}
  </script>
  <script type="module">
    import * as THREE from "three";
    import {{ OrbitControls }} from "three/addons/controls/OrbitControls.js";
    import {{ STLLoader }} from "three/addons/loaders/STLLoader.js";
    import {{ ColladaLoader }} from "three/addons/loaders/ColladaLoader.js";

    const data = {payload_json};
    const root = document.getElementById("viewer");
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf6f6f4);

    const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.01, 100);
    camera.position.set(4.4, 3.4, 6.2);
    camera.lookAt(0, 0.7, 0);

    const renderer = new THREE.WebGLRenderer({{ antialias: true }});
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight);
    root.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0.7, 0);
    controls.update();

    scene.add(new THREE.HemisphereLight(0xffffff, 0x888888, 2.0));
    const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
    dirLight.position.set(3, 5, 4);
    scene.add(dirLight);

    const grid = new THREE.GridHelper(7, 28, 0xb8c2cc, 0xd9e2ec);
    scene.add(grid);

    const matBase = new THREE.MeshStandardMaterial({{ color: 0x4c72b0, transparent: true, opacity: 0.78 }});
    const matArm = new THREE.MeshStandardMaterial({{ color: 0x1f2933 }});
    const matEE = new THREE.MeshStandardMaterial({{ color: 0x55a868 }});
    const matObstacle = new THREE.MeshStandardMaterial({{ color: 0xc44e52, transparent: true, opacity: 0.32 }});
    const matGoal = new THREE.MeshStandardMaterial({{ color: 0x55a868, emissive: 0x174f2a, emissiveIntensity: 0.15 }});
    const matRobot = new THREE.MeshStandardMaterial({{ color: 0x9aa5b1, metalness: 0.05, roughness: 0.62, side: THREE.DoubleSide }});
    const matProxy = new THREE.MeshStandardMaterial({{
      color: 0xf0a202,
      transparent: true,
      opacity: 0.23,
      depthWrite: false,
      roughness: 0.75,
    }});
    const robotVisuals = [];
    const collisionProxyVisuals = [];
    const frameAxes = [];
    const frameTraces = [];
    const trackedFrames = ["mobile_base", "link1", "link2", "link3", "link4", "link5", "link6", "gripper_base"];
    const frameTraceColors = {{
      mobile_base: 0x4c72b0,
      link1: 0xdd8452,
      link2: 0x55a868,
      link3: 0xc44e52,
      link4: 0x8172b3,
      link5: 0x937860,
      link6: 0xda8bc3,
      gripper_base: 0x2f7d43,
    }};
    const stlLoader = new STLLoader();
    const colladaLoader = new ColladaLoader();

    for (const obs of data.obstacles) {{
      let mesh;
      if (obs.type === "sphere") {{
        mesh = new THREE.Mesh(new THREE.SphereGeometry(obs.radius, 32, 16), matObstacle);
        mesh.position.copy(toThree(obs.center));
      }} else if (obs.type === "cuboid") {{
        mesh = new THREE.Mesh(new THREE.BoxGeometry(obs.size[0], obs.size[2], obs.size[1]), matObstacle);
        mesh.position.copy(toThree(obs.center));
      }} else {{
        mesh = new THREE.Mesh(new THREE.CylinderGeometry(obs.radius, obs.radius, obs.height, 32), matObstacle);
        mesh.position.copy(toThree(obs.center));
      }}
      scene.add(mesh);
    }}

    data.goals.forEach((g, i) => {{
      const goal = new THREE.Mesh(new THREE.SphereGeometry(0.07, 20, 12), matGoal);
      goal.position.copy(toThree(g));
      scene.add(goal);
      const label = makeLabel(`g${{i + 1}}`);
      label.position.copy(toThree([g[0] + 0.06, g[1] + 0.06, g[2] + 0.08]));
      scene.add(label);
    }});

    const base = new THREE.Mesh(new THREE.CylinderGeometry(data.baseRadius, data.baseRadius, data.baseHeight, 32), matBase);
    if (!hasRobotVisuals()) scene.add(base);
    const heading = makeLine(0x4c72b0);
    scene.add(heading);
    heading.visible = false;

    const armChain = new THREE.Line(
      new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({{ color: 0x1f2933, linewidth: 3 }})
    );
    scene.add(armChain);
    const mountJoint = makeSphere(0.045, 0x1f2933);
    const elbowJoint = makeSphere(0.055, 0x1f2933);
    const eeJoint = makeSphere(0.065, 0x55a868);
    scene.add(mountJoint, elbowJoint, eeJoint);
    if (hasRobotVisuals()) {{
      mountJoint.visible = false;
      elbowJoint.visible = false;
      eeJoint.visible = false;
      armChain.visible = false;
    }}
    loadRobotVisuals();
    loadCollisionProxies();

    const baseTrace = makeTrace(0x4c72b0);
    const eeTrace = makeTrace(0x55a868);
    scene.add(baseTrace, eeTrace);
    if (hasRobotVisuals()) {{
      baseTrace.visible = false;
      eeTrace.visible = false;
    }}
    buildFrameMarkers();

    const slider = document.getElementById("slider");
    const label = document.getElementById("stepLabel");
    const infoPanel = document.getElementById("infoPanel");
    const play = document.getElementById("play");
    const speed = document.getElementById("speed");
    const showProxies = document.getElementById("showProxies");
    slider.max = data.frames.length - 1;
    const metadata = data.metadata || {{}};
    const minClearance = Math.min(...data.frames.map(f => f.clearance));

    let current = 0;
    let playing = false;
    let lastTick = 0;

    function update(step) {{
      current = Math.max(0, Math.min(data.frames.length - 1, step));
      slider.value = current;
      const frame = data.frames[current];
      const segment = data.segmentEnds.findIndex(end => current <= end);
      const target = segment < 0 ? data.goals.length : segment + 1;
      label.textContent = `step ${{current}} / ${{data.frames.length - 1}} | target g${{target}}`;
      updateInfoPanel(frame, target);

      base.position.copy(toThree([frame.base[0], frame.base[1], data.baseHeight / 2]));
      base.rotation.y = -frame.yaw;
      updateRobotVisuals(frame);
      updateCollisionProxies(frame);
      updateFrameMarkers(frame);
      setLine(heading, frame.base, [
        frame.base[0] + data.baseRadius * Math.cos(frame.yaw),
        frame.base[1] + data.baseRadius * Math.sin(frame.yaw),
        data.baseHeight * 0.5,
      ]);
      armChain.geometry.setFromPoints((frame.points || [frame.mount, frame.elbow, frame.ee]).map(toThree));
      mountJoint.position.copy(toThree(frame.mount));
      elbowJoint.position.copy(toThree(frame.elbow));
      eeJoint.position.copy(toThree(frame.ee));

      const basePoints = data.frames.slice(0, current + 1).map(f => toThree([f.base[0], f.base[1], 0.02]));
      const eePoints = data.frames.slice(0, current + 1).map(f => toThree(f.ee));
      baseTrace.geometry.setFromPoints(basePoints);
      eeTrace.geometry.setFromPoints(eePoints);
      updateFrameTraces(current);
    }}

    function updateInfoPanel(frame, target) {{
      const state = frame.state || [frame.base[0], frame.base[1], frame.yaw, 0, 0, 0];
      const goal = data.goals[Math.max(0, Math.min(data.goals.length - 1, target - 1))];
      const eeError = goal ? distance(frame.ee, goal) : 0;
      const planner = metadata.planner || "unknown";
      const status = metadata.success === false ? "failed" : "ok";
      const targetIndex = Math.max(0, Math.min(data.goals.length - 1, target - 1));
      const iterations = Array.isArray(metadata.iterations) ? metadata.iterations[targetIndex] : undefined;
      const planningTime = Array.isArray(metadata.planningTimes) ? metadata.planningTimes[targetIndex] : undefined;
      const sceneBits = [];
      if (metadata.seed !== undefined) sceneBits.push(`seed ${{metadata.seed}}`);
      if (metadata.obstacleCount !== undefined) sceneBits.push(`obs ${{metadata.obstacleCount}}`);
      const planBits = [];
      if (iterations !== undefined) planBits.push(`${{iterations}} it`);
      if (planningTime !== undefined) planBits.push(`${{fmt(planningTime)}} s`);
      infoPanel.innerHTML = `
        <div class="title"><span>${{escapeHtml(planner)}}</span><span>${{status}}</span></div>
        <div class="grid">
          <div class="key">frame</div><div class="value">${{current}} / ${{data.frames.length - 1}} · g${{target}}</div>
          <div class="key">scene</div><div class="value">${{escapeHtml(sceneBits.join(" · ") || "-")}}</div>
          <div class="key">plan</div><div class="value">${{escapeHtml(planBits.join(" · ") || "-")}}</div>
          <div class="key">base</div><div class="value">x ${{fmt(state[0])}}, y ${{fmt(state[1])}}, yaw ${{fmtRad(state[2])}}</div>
          <div class="key">arm</div><div class="value">q ${{fmtRad(state[3])}}, ${{fmtRad(state[4])}}, ${{fmtRad(state[5])}}</div>
          <div class="key">ee error</div><div class="value">${{fmt(eeError)}} m</div>
          <div class="key">clearance</div><div class="value">${{fmt(frame.clearance)}} m · min ${{fmt(minClearance)}} m</div>
        </div>
      `;
    }}

    slider.addEventListener("input", () => update(Number(slider.value)));
    showProxies.addEventListener("change", () => updateCollisionProxies(data.frames[current]));
    play.addEventListener("click", () => {{
      playing = !playing;
      play.textContent = playing ? "Pause" : "Play";
    }});

    function animate(time) {{
      requestAnimationFrame(animate);
      const frameIntervalMs = 70 / Number(speed.value);
      if (playing && time - lastTick > frameIntervalMs) {{
        lastTick = time;
        if (current >= data.frames.length - 1) {{
          playing = false;
          play.textContent = "Play";
        }} else {{
          update(current + 1);
        }}
      }}
      controls.update();
      renderer.render(scene, camera);
    }}

    window.addEventListener("resize", () => {{
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    }});

    function hasRobotVisuals() {{
      return Array.isArray(data.robotVisuals) && data.robotVisuals.length > 0;
    }}

    function loadRobotVisuals() {{
      if (!hasRobotVisuals()) return;
      for (const spec of data.robotVisuals) {{
        if (spec.type === "box") {{
          const mesh = new THREE.Mesh(
            new THREE.BoxGeometry(spec.size[0], spec.size[1], spec.size[2]),
            matRobot.clone()
          );
          scene.add(mesh);
          robotVisuals.push({{ spec, object: mesh }});
        }} else if (spec.type === "mesh" && spec.format === "stl") {{
          stlLoader.load(spec.url, geometry => {{
            geometry.computeVertexNormals();
            const mesh = new THREE.Mesh(geometry, matRobot.clone());
            scene.add(mesh);
            robotVisuals.push({{ spec, object: mesh }});
            update(current);
          }}, undefined, error => {{
            console.error("Failed to load robot STL mesh", spec.link, spec.url, error);
          }});
        }} else if (spec.type === "mesh" && spec.format === "dae") {{
          colladaLoader.load(spec.url, collada => {{
            const object = collada.scene;
            object.traverse(child => {{
              if (child.isMesh) child.material = matRobot.clone();
            }});
            scene.add(object);
            robotVisuals.push({{ spec, object }});
            update(current);
          }}, undefined, error => {{
            console.error("Failed to load robot DAE mesh", spec.link, spec.url, error);
          }});
        }}
      }}
    }}

    function updateRobotVisuals(frame) {{
      if (!hasRobotVisuals() || !frame.linkTransforms) return;
      for (const item of robotVisuals) {{
        const linkMatrix = frame.linkTransforms[item.spec.link];
        if (!linkMatrix) continue;
        const worldMatrix = multiplyMatrix4(linkMatrix, item.spec.originMatrix);
        setObjectMatrixFromRobotMatrix(item.object, worldMatrix);
      }}
    }}

    function loadCollisionProxies() {{
      if (!Array.isArray(data.collisionProxies)) return;
      for (const spec of data.collisionProxies) {{
        let object;
        if (spec.type === "box") {{
          object = new THREE.Mesh(new THREE.BoxGeometry(spec.size[0], spec.size[1], spec.size[2]), matProxy.clone());
        }} else if (spec.type === "sphere") {{
          object = new THREE.Mesh(new THREE.SphereGeometry(spec.radius, 24, 12), matProxy.clone());
        }} else if (spec.type === "capsule") {{
          object = new THREE.Mesh(new THREE.CapsuleGeometry(spec.radius, spec.length, 8, 16), matProxy.clone());
        }} else {{
          continue;
        }}
        object.renderOrder = 3;
        scene.add(object);
        collisionProxyVisuals.push({{ spec, object }});
      }}
    }}

    function updateCollisionProxies(frame) {{
      if (!frame.linkTransforms) return;
      const visible = showProxies.checked;
      for (const item of collisionProxyVisuals) {{
        const linkMatrix = frame.linkTransforms[item.spec.link];
        if (!visible || !linkMatrix) {{
          item.object.visible = false;
          continue;
        }}
        item.object.visible = true;
        const worldMatrix = multiplyMatrix4(linkMatrix, item.spec.originMatrix);
        setObjectMatrixFromRobotMatrix(item.object, worldMatrix);
      }}
    }}

    function buildFrameMarkers() {{
      for (const name of trackedFrames) {{
        const axes = makeFrameAxes(frameLabel(name), name === "mobile_base" ? 0.24 : 0.14);
        axes.name = name;
        axes.visible = false;
        scene.add(axes);
        frameAxes.push({{ name, object: axes }});

        const trace = makeTrace(frameTraceColors[name] || 0x6f6f6f);
        trace.visible = true;
        scene.add(trace);
        frameTraces.push({{ name, object: trace }});
      }}
    }}

    function updateFrameMarkers(frame) {{
      if (!frame.linkTransforms) return;
      for (const item of frameAxes) {{
        const matrix = frame.linkTransforms[item.name];
        if (!matrix) {{
          item.object.visible = false;
          continue;
        }}
        item.object.visible = true;
        setObjectMatrixFromRobotMatrix(item.object, matrix);
      }}
    }}

    function updateFrameTraces(step) {{
      for (const item of frameTraces) {{
        const points = [];
        for (let i = 0; i <= step; i++) {{
          const origin = data.frames[i].frameOrigins && data.frames[i].frameOrigins[item.name];
          if (origin) points.push(toThree(origin));
        }}
        item.object.geometry.setFromPoints(points);
      }}
    }}

    function makeFrameAxes(label, length) {{
      const group = new THREE.Group();
      group.add(makeAxisLine([0, 0, 0], [length, 0, 0], 0xd62728));
      group.add(makeAxisLine([0, 0, 0], [0, length, 0], 0x2ca02c));
      group.add(makeAxisLine([0, 0, 0], [0, 0, length], 0x1f77b4));
      const tag = makeSmallLabel(label);
      tag.position.set(0, 0, length * 1.25);
      group.add(tag);
      return group;
    }}

    function makeAxisLine(a, b, color) {{
      const line = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(...a), new THREE.Vector3(...b)]),
        new THREE.LineBasicMaterial({{ color, linewidth: 2 }})
      );
      return line;
    }}

    function makeLine(color) {{
      const mat = new THREE.LineBasicMaterial({{ color, linewidth: 3 }});
      return new THREE.Line(new THREE.BufferGeometry(), mat);
    }}

    function makeTrace(color) {{
      const mat = new THREE.LineBasicMaterial({{ color, transparent: true, opacity: 0.85 }});
      return new THREE.Line(new THREE.BufferGeometry(), mat);
    }}

    function setLine(line, a, b) {{
      line.geometry.setFromPoints([toThree(a), toThree(b)]);
    }}

    function makeSphere(radius, color) {{
      return new THREE.Mesh(new THREE.SphereGeometry(radius, 20, 12), new THREE.MeshStandardMaterial({{ color }}));
    }}

    function makeLabel(text) {{
      const canvas = document.createElement("canvas");
      canvas.width = 96;
      canvas.height = 48;
      const ctx = canvas.getContext("2d");
      ctx.fillStyle = "rgba(255,255,255,0.85)";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#2f7d43";
      ctx.font = "24px sans-serif";
      ctx.fillText(text, 18, 31);
      const texture = new THREE.CanvasTexture(canvas);
      const material = new THREE.SpriteMaterial({{ map: texture, transparent: true }});
      const sprite = new THREE.Sprite(material);
      sprite.scale.set(0.22, 0.11, 1);
      return sprite;
    }}

    function makeSmallLabel(text) {{
      const canvas = document.createElement("canvas");
      canvas.width = 128;
      canvas.height = 48;
      const ctx = canvas.getContext("2d");
      ctx.fillStyle = "rgba(255,255,255,0.72)";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#1f2933";
      ctx.font = "22px sans-serif";
      ctx.fillText(text, 10, 31);
      const texture = new THREE.CanvasTexture(canvas);
      const material = new THREE.SpriteMaterial({{ map: texture, transparent: true }});
      const sprite = new THREE.Sprite(material);
      sprite.scale.set(0.16, 0.06, 1);
      return sprite;
    }}

    function frameLabel(name) {{
      if (name === "mobile_base") return "base";
      if (name === "gripper_base") return "ee";
      if (name.startsWith("link")) return `j${{name.slice(4)}}`;
      return name;
    }}

    function toThree(p) {{
      // Robot/world coordinates are Z-up: [x, y, z].
      // Three.js is Y-up, so render as [x, z, -y].
      return new THREE.Vector3(p[0], p[2], -p[1]);
    }}

    function setObjectMatrixFromRobotMatrix(object, matrix) {{
      const converted = robotMatrixToThreeMatrix(matrix);
      object.matrixAutoUpdate = false;
      object.matrix.copy(converted);
      object.matrixWorldNeedsUpdate = true;
    }}

    function robotMatrixToThreeMatrix(m) {{
      const r = new THREE.Matrix4().set(
        m[0][0], m[0][1], m[0][2], m[0][3],
        m[1][0], m[1][1], m[1][2], m[1][3],
        m[2][0], m[2][1], m[2][2], m[2][3],
        m[3][0], m[3][1], m[3][2], m[3][3],
      );
      const c = new THREE.Matrix4().set(
        1, 0, 0, 0,
        0, 0, 1, 0,
        0, -1, 0, 0,
        0, 0, 0, 1,
      );
      return c.multiply(r);
    }}

    function multiplyMatrix4(a, b) {{
      const out = Array.from({{ length: 4 }}, () => [0, 0, 0, 0]);
      for (let i = 0; i < 4; i++) {{
        for (let j = 0; j < 4; j++) {{
          for (let k = 0; k < 4; k++) out[i][j] += a[i][k] * b[k][j];
        }}
      }}
      return out;
    }}

    function distance(a, b) {{
      return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    }}

    function fmt(value) {{
      return Number.isFinite(value) ? value.toFixed(3) : "-";
    }}

    function fmtRad(value) {{
      return Number.isFinite(value) ? value.toFixed(2) : "-";
    }}

    function escapeHtml(value) {{
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }}

    update(0);
    animate(0);
  </script>
</body>
</html>
"""
