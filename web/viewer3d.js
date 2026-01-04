const THREE_URL = "https://esm.sh/three@0.160.0";
const FBX_LOADER_URL = "https://esm.sh/three@0.160.0/examples/jsm/loaders/FBXLoader.js";
const ORBIT_CONTROLS_URL = "https://esm.sh/three@0.160.0/examples/jsm/controls/OrbitControls.js";
const GLTF_LOADER_URL = "https://esm.sh/three@0.160.0/examples/jsm/loaders/GLTFLoader.js";
const OBJ_LOADER_URL = "https://esm.sh/three@0.160.0/examples/jsm/loaders/OBJLoader.js";
const GLTF_EXPORTER_URL = "https://esm.sh/three@0.160.0/examples/jsm/exporters/GLTFExporter.js";

import { app } from "../../../scripts/app.js";

console.log("[HY-Motion] app imported successfully:", !!app);

app.registerExtension({
    name: "HYMotion.3DViewer",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "HYMotion3DViewer" &&
            nodeData.name !== "HYMotionFBXPlayer" &&
            nodeData.name !== "HYMotion3DModelLoader") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            // Main container
            const container = document.createElement("div");
            container.style.cssText = "width:100%; height:400px; background:#111; position:relative; display:flex; flex-direction:column; border:1px solid #333; border-radius:4px; overflow:hidden;";

            const canvasContainer = document.createElement("div");
            canvasContainer.style.cssText = "flex:1; width:100%;";
            container.appendChild(canvasContainer);

            // Playback controls
            const controls = document.createElement("div");
            controls.style.cssText = "height:30px; display:flex; align-items:center; padding:0 10px; gap:10px; background:#222;";

            const playBtn = document.createElement("button");
            playBtn.innerText = "Play";
            playBtn.style.cssText = "cursor:pointer; padding:2px 10px;";

            const progress = document.createElement("input");
            progress.type = "range";
            progress.min = 0;
            progress.max = 100;
            progress.value = 0;
            progress.style.flex = "1";

            const statusLabel = document.createElement("div");
            statusLabel.style.cssText = "font-size:11px; color:#888;";
            statusLabel.innerText = "Ready";

            const cycleBtn = document.createElement("button");
            cycleBtn.innerText = "Select";
            cycleBtn.title = "Cycle through loaded models and skeletons";
            cycleBtn.style.cssText = "cursor:pointer; padding:2px 6px; font-size:11px; background:#444; color:#fff; border:1px solid #666; border-radius:3px;";

            const exportBtn = document.createElement("button");
            exportBtn.innerText = "Export";
            exportBtn.title = "Export or Download selection";
            exportBtn.style.cssText = "cursor:pointer; padding:2px 6px; font-size:11px; background:#226622; color:#fff; border:1px solid #338833; border-radius:3px; display:none;";

            controls.appendChild(playBtn);
            controls.appendChild(cycleBtn);
            controls.appendChild(exportBtn);
            controls.appendChild(progress);
            controls.appendChild(statusLabel);
            container.appendChild(controls);

            this.addDOMWidget("3d_viewer", "viewer", container);
            this.size = [400, 480];

            let THREE = window.__HY_MOTION_THREE__ || null;
            let renderer, scene, camera, orbitControls;
            let currentModel = null, mixer = null;
            let skeletalSamples = [];
            let isPlaying = false;
            let clock = null;
            let currentFrame = 0;
            let maxFrames = 0;
            let frameAccumulator = 0;
            const targetFPS = 30;
            const frameTime = 1 / targetFPS;
            let mixers = [];
            let isInitialized = false;
            let pendingDataQueue = [];
            const node = this;

            let raycaster = null;
            let selectedModels = []; // Array to support multi-selection
            let loadedModels = [];

            const SMPL_H_SKELETON = [
                [0, 1], [0, 2], [0, 3],
                [1, 4], [2, 5], [3, 6],
                [4, 7], [5, 8], [6, 9],
                [7, 10], [8, 11], [9, 12],
                [9, 13], [9, 14], // Shoulders connect to Spine3 (9), not Neck (12)
                [12, 15], // Neck to Head
                [13, 16], [14, 17],
                [16, 18], [17, 19],
                [18, 20], [19, 21],
                // Left Fingers
                [20, 22], [22, 23], [23, 24], // Index
                [20, 25], [25, 26], [26, 27], // Middle
                [20, 28], [28, 29], [29, 30], // Pinky
                [20, 31], [31, 32], [32, 33], // Ring
                [20, 34], [34, 35], [35, 36], // Thumb
                // Right Fingers
                [21, 37], [37, 38], [38, 39], // Index
                [21, 40], [40, 41], [41, 42], // Middle
                [21, 43], [43, 44], [44, 45], // Pinky
                [21, 46], [46, 47], [47, 48], // Ring
                [21, 49], [49, 50], [50, 51]  // Thumb
            ];

            const initThree = async () => {
                try {
                    if (!window.__HY_MOTION_THREE__) {
                        console.log("[HY-Motion] Loading Three.js libs...");
                        window.__HY_MOTION_THREE__ = await import(THREE_URL);
                        window.__HY_MOTION_ORBIT__ = await import(ORBIT_CONTROLS_URL);
                        window.__HY_MOTION_EXPORTER__ = await import(GLTF_EXPORTER_URL);
                    }
                    THREE = window.__HY_MOTION_THREE__;
                    const { OrbitControls } = window.__HY_MOTION_ORBIT__;
                    const { GLTFExporter } = window.__HY_MOTION_EXPORTER__;

                    if (!scene) {
                        scene = new THREE.Scene();
                        scene.background = new THREE.Color(0x111111);

                        camera = new THREE.PerspectiveCamera(50, 1, 0.1, 1000);
                        camera.position.set(0, 2, 5);

                        renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
                        const width = canvasContainer.clientWidth || 400;
                        const height = canvasContainer.clientHeight || 400;
                        renderer.setSize(width, height);
                        camera.aspect = width / height;
                        camera.updateProjectionMatrix();

                        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
                        canvasContainer.appendChild(renderer.domElement);

                        orbitControls = new OrbitControls(camera, renderer.domElement);
                        orbitControls.enableDamping = true;
                        orbitControls.dampingFactor = 0.05;
                        orbitControls.target.set(0, 0.8, 0);

                        // Allow middle mouse to rotate (like Blender)
                        orbitControls.mouseButtons = {
                            LEFT: THREE.MOUSE.ROTATE,
                            MIDDLE: THREE.MOUSE.ROTATE,
                            RIGHT: THREE.MOUSE.PAN
                        };

                        scene.add(new THREE.AmbientLight(0xffffff, 1.0));
                        const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
                        dirLight.position.set(5, 10, 7.5);
                        scene.add(dirLight);

                        const grid = new THREE.GridHelper(10, 10, 0x666666, 0x333333);
                        scene.add(grid);

                        const originMarker = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.05, 0.05), new THREE.MeshBasicMaterial({ color: 0xff0000 }));
                        scene.add(originMarker);

                        const groundGeo = new THREE.PlaneGeometry(20, 20);
                        const groundMat = new THREE.MeshStandardMaterial({ color: 0x0a0a0a });
                        const ground = new THREE.Mesh(groundGeo, groundMat);
                        ground.rotation.x = -Math.PI / 2;
                        ground.position.y = -0.01;
                        scene.add(ground);

                        clock = new THREE.Clock();

                        const resizeObserver = new ResizeObserver(() => {
                            if (canvasContainer.clientWidth > 0 && canvasContainer.clientHeight > 0) {
                                camera.aspect = canvasContainer.clientWidth / canvasContainer.clientHeight;
                                camera.updateProjectionMatrix();
                                renderer.setSize(canvasContainer.clientWidth, canvasContainer.clientHeight);
                            }
                        });
                        resizeObserver.observe(canvasContainer);

                        raycaster = new THREE.Raycaster();
                        canvasContainer.addEventListener('pointerdown', onCanvasPointerDown);
                        canvasContainer.addEventListener('pointermove', onCanvasPointerMove);
                    }

                    isInitialized = true;
                    while (pendingDataQueue.length > 0) handleData(pendingDataQueue.shift());
                    animate();
                } catch (e) {
                    console.error("[HY-Motion] Viewer Init Error:", e);
                    statusLabel.innerText = "Error: Three.js fail";
                }
            };

            const getIntersectables = () => {
                const intersectableObjects = [];
                for (const obj of loadedModels) {
                    if (obj.hitProxy) intersectableObjects.push(obj.hitProxy);
                }
                for (const sample of skeletalSamples) {
                    for (const joint of sample.joints) {
                        intersectableObjects.push(joint);
                    }
                }
                return intersectableObjects;
            };

            const onCanvasPointerMove = (event) => {
                if (!raycaster || !camera || !scene) return;
                const rect = renderer.domElement.getBoundingClientRect();
                const mouse = new THREE.Vector2();
                mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
                mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
                raycaster.setFromCamera(mouse, camera);

                // Check intersects using proxies/joints
                const intersects = raycaster.intersectObjects(getIntersectables(), false);
                renderer.domElement.style.cursor = intersects.length > 0 ? "pointer" : "auto";
            };

            const onCanvasPointerDown = (event) => {
                if (!raycaster || !camera || !scene) return;

                // Prevent interfering with orbit controls mouse/touch
                if (event.button !== 0) return;

                const rect = renderer.domElement.getBoundingClientRect();
                const mouse = new THREE.Vector2();
                mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
                mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

                raycaster.setFromCamera(mouse, camera);
                const intersects = raycaster.intersectObjects(getIntersectables(), false);

                if (intersects.length > 0) {
                    // Find first valid hit
                    let hit = null;
                    for (let i = 0; i < intersects.length; i++) {
                        if (intersects[i].object.userData.selectableParent) {
                            hit = intersects[i].object;
                            break;
                        }
                    }

                    if (hit) {
                        selectObject(hit.userData.selectableParent, hit.userData.type, event.ctrlKey);
                        event.stopPropagation();
                        // DO NOT preventDefault here as it might break OrbitControls
                    }
                } else {
                    if (!event.ctrlKey) deselectAll();
                }
            };


            const cycleSelection = () => {
                const all = [
                    ...loadedModels.map(m => ({ obj: m, type: "model" })),
                    ...skeletalSamples.map(s => ({ obj: s, type: "skeleton" }))
                ];
                if (all.length === 0) return;

                let nextIdx = 0;
                if (selectedModels.length > 0) {
                    const lastSelected = selectedModels[selectedModels.length - 1];
                    const currentIdx = all.findIndex(item => item.obj === lastSelected.obj);
                    nextIdx = (currentIdx + 1) % all.length;
                }
                const next = all[nextIdx];
                deselectAll(); // Cycle clears others for clarity
                selectObject(next.obj, next.type, false);
            };

            const getRealtimeBox = (obj) => {
                const box = new THREE.Box3();
                obj.traverse((child) => {
                    if (child.isMesh) {
                        if (child.isSkinnedMesh && child.skeleton) {
                            // Find bounds of all bones in world space as an approximation of the volume
                            const boneBox = new THREE.Box3();
                            child.skeleton.bones.forEach(bone => {
                                boneBox.expandByPoint(bone.getWorldPosition(new THREE.Vector3()));
                            });
                            // Buff slightly to cover the mesh thickness
                            boneBox.expandByScalar(0.15);
                            box.union(boneBox);
                        } else {
                            const childBox = new THREE.Box3().setFromObject(child);
                            if (!childBox.isEmpty()) box.union(childBox);
                        }
                    }
                });
                return box;
            };

            const selectObject = (obj, type, isMulti = false) => {
                if (!isMulti) deselectAll();

                // Toggle if already selected
                const existingIdx = selectedModels.findIndex(m => m.obj === obj);
                if (existingIdx !== -1) {
                    if (isMulti) {
                        removeSelectionAt(existingIdx);
                        updateUI();
                        return;
                    } else {
                        // If clicking same without multi, just keep it (deselect others happened above)
                    }
                } else {
                    const selection = { obj, type };
                    applyHighlight(selection);
                    selectedModels.push(selection);
                }
                updateUI();
            };

            const applyHighlight = (selection) => {
                const { obj, type } = selection;
                if (type === "model") {
                    const target = obj.model;
                    target.traverse((child) => {
                        if (child.isMesh) {
                            const mats = Array.isArray(child.material) ? child.material : [child.material];
                            for (const m of mats) {
                                if (!m) continue;
                                if (!m.userData) m.userData = {};
                                if (!m.userData.origEmissive) m.userData.origEmissive = m.emissive ? m.emissive.clone() : new THREE.Color(0, 0, 0);
                                m.emissive = new THREE.Color(0x00ff00);
                                m.emissiveIntensity = 0.6;
                            }
                        }
                    });

                    const box = new THREE.BoxHelper(new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1)), 0x00ff00);
                    box.name = "selection_highlight";
                    scene.add(box);
                    selection.highlight = box;

                } else if (type === "skeleton") {
                    for (const joint of obj.joints) {
                        joint.material.emissive = new THREE.Color(0xffff00);
                        joint.material.emissiveIntensity = 1.0;
                    }

                    const box = new THREE.BoxHelper(new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1)), 0xffff00);
                    box.name = "selection_highlight";
                    scene.add(box);
                    selection.highlight = box;
                }
            };

            const updateUI = () => {
                const count = selectedModels.length;
                const isLegacyPlayer = nodeData.name === "HYMotionFBXPlayer";

                if (count === 0) {
                    statusLabel.innerText = "Ready";
                    statusLabel.style.color = "#888";
                    exportBtn.style.display = "none";
                } else if (count === 1) {
                    const s = selectedModels[0];
                    statusLabel.innerText = `Selected ${s.type === 'model' ? 'Model' : 'Skeleton'}: ${s.obj.name || ''}`;
                    statusLabel.style.color = "#0f0";

                    if (isLegacyPlayer) {
                        exportBtn.style.display = "block";
                        exportBtn.innerText = "Download FBX";
                        exportBtn.title = "Download the original FBX file";
                    } else {
                        exportBtn.style.display = "block";
                        exportBtn.innerText = "Export GLB";
                        exportBtn.title = "Export selection as GLB (3D Mesh)";
                    }
                } else {
                    statusLabel.innerText = `Selected: ${count} objects (Ctrl+Click to add)`;
                    statusLabel.style.color = "#0f0";
                    exportBtn.style.display = isLegacyPlayer ? "none" : "block"; // Only supports single download for legacy player
                    if (!isLegacyPlayer) {
                        exportBtn.innerText = `Export ${count} GLBs`;
                    }
                }
            };

            const removeSelectionAt = (idx) => {
                const selection = selectedModels[idx];
                if (selection.highlight) scene.remove(selection.highlight);

                if (selection.type === "model") {
                    selection.obj.model.traverse((child) => {
                        if (child.isMesh) {
                            const mats = Array.isArray(child.material) ? child.material : [child.material];
                            for (const m of mats) {
                                if (!m) continue;
                                if (m.userData && m.userData.origEmissive) m.emissive.copy(m.userData.origEmissive);
                                else m.emissive = new THREE.Color(0, 0, 0);
                                m.emissiveIntensity = 0;
                            }
                        }
                    });
                } else if (selection.type === "skeleton") {
                    for (const joint of selection.obj.joints) {
                        joint.material.emissiveIntensity = 0.3;
                    }
                }
                selectedModels.splice(idx, 1);
            };

            const focusOn = (obj) => {
                if (!obj || !orbitControls) return;
                const box = new THREE.Box3().setFromObject(obj);
                const center = box.getCenter(new THREE.Vector3());
                const size = box.getSize(new THREE.Vector3());
                const maxDim = Math.max(size.x, size.y, size.z);
                const targetPos = new THREE.Vector3(center.x, center.y + (maxDim * 0.2), center.z + Math.max(maxDim * 2.5, 3));

                const startPos = camera.position.clone();
                const startTarget = orbitControls.target.clone();
                let t = 0;
                const anim = () => {
                    t += 0.05;
                    if (t <= 1.0) {
                        camera.position.lerpVectors(startPos, targetPos, t);
                        orbitControls.target.lerpVectors(startTarget, center, t);
                        requestAnimationFrame(anim);
                    }
                };
                anim();
            };

            const deselectAll = () => {
                while (selectedModels.length > 0) {
                    removeSelectionAt(0);
                }
                updateUI();
            };

            const exportSelected = async () => {
                if (selectedModels.length === 0) {
                    alert("Please select at least one object to export.");
                    return;
                }

                if (nodeData.name === "HYMotionFBXPlayer") {
                    // Direct Download Logic for Legacy FBX Player
                    const selection = selectedModels[0];
                    if (selection.type !== "model") return;

                    const { fileName, subfolder, fileType } = selection.obj;

                    let fetchUrl = `${window.location.origin}/view?type=${encodeURIComponent(fileType || 'output')}&filename=${encodeURIComponent(fileName)}`;
                    if (subfolder) fetchUrl += `&subfolder=${encodeURIComponent(subfolder)}`;

                    const link = document.createElement('a');
                    link.href = fetchUrl;
                    link.download = fileName;
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                    return;
                }

                const { GLTFExporter } = window.__HY_MOTION_EXPORTER__;
                if (!GLTFExporter) {
                    alert("GLTFExporter not loaded yet.");
                    return;
                }

                const exporter = new GLTFExporter();

                for (let i = 0; i < selectedModels.length; i++) {
                    const selection = selectedModels[i];
                    let objToExport;
                    if (selection.type === "model") objToExport = selection.obj.model;
                    else if (selection.type === "skeleton") objToExport = selection.obj.group;

                    if (!objToExport) continue;

                    let filename = (selection.obj.name || (selection.type + "_" + i)) + ".glb";
                    console.log(`[HY-Motion] Exporting ${i + 1}/${selectedModels.length}:`, filename);

                    // Clone to reset highlights for export
                    const clone = objToExport.clone();
                    clone.traverse(child => {
                        if (child.isMesh && child.material) {
                            const mats = Array.isArray(child.material) ? child.material : [child.material];
                            mats.forEach(m => {
                                if (m.userData && m.userData.origEmissive) m.emissive.copy(m.userData.origEmissive);
                                else if (m.emissive) m.emissive.set(0, 0, 0);
                                m.emissiveIntensity = 0;
                            });
                        }
                    });

                    // Trigger export
                    await new Promise((resolve) => {
                        exporter.parse(
                            clone,
                            (result) => {
                                const blob = new Blob([result], { type: 'application/octet-stream' });
                                const link = document.createElement('a');
                                link.href = URL.createObjectURL(blob);
                                link.download = filename;
                                link.click();
                                setTimeout(resolve, 300);
                            },
                            (error) => {
                                console.error("[HY-Motion] Export failed:", error);
                                resolve();
                            },
                            { binary: true, animations: [] }
                        );
                    });
                }
                console.log("[HY-Motion] Multi-export sequence complete.");
            };

            const animate = () => {
                requestAnimationFrame(animate);
                if (!THREE) return; // Guard against early frames
                const delta = clock ? clock.getDelta() : 0.016;
                if (isPlaying) {
                    if (mixers.length > 0) {
                        for (const m of mixers) m.update(delta);
                        if (maxFrames > 0 && mixers[0]) {
                            const time = mixers[0].time % (maxFrames * frameTime);
                            currentFrame = Math.floor(time * targetFPS);
                            progress.value = (currentFrame / maxFrames) * 100;
                        }
                    } else if (maxFrames > 0) {
                        frameAccumulator += delta;
                        while (frameAccumulator >= frameTime) {
                            currentFrame = (currentFrame + 1) % maxFrames;
                            frameAccumulator -= frameTime;
                        }
                        updateSkeletons(currentFrame);
                        progress.value = (currentFrame / maxFrames) * 100;
                    }
                }

                // Update hit proxies every frame to track animated movement
                for (const obj of loadedModels) {
                    if (!obj.hitProxy && THREE) {
                        const geo = new THREE.BoxGeometry(1, 1, 1);
                        const mat = new THREE.MeshBasicMaterial({
                            visible: false,
                            wireframe: true,
                            side: THREE.DoubleSide // ALLOW SELECTING FROM INSIDE
                        });
                        const proxy = new THREE.Mesh(geo, mat);
                        proxy.userData.selectableParent = obj;
                        proxy.userData.type = "model";
                        scene.add(proxy);
                        obj.hitProxy = proxy;
                    }

                    if (obj.hitProxy) {
                        const box = getRealtimeBox(obj.model);
                        if (!box.isEmpty()) {
                            const size = box.getSize(new THREE.Vector3());
                            const center = box.getCenter(new THREE.Vector3());
                            obj.hitProxy.scale.set(size.x * 1.3, size.y * 1.1, size.z * 1.3);
                            obj.hitProxy.position.copy(center);
                        }
                    }
                }

                for (const selection of selectedModels) {
                    if (selection.highlight) {
                        if (selection.type === "model") {
                            const box = getRealtimeBox(selection.obj.model);
                            if (!box.isEmpty()) {
                                const size = box.getSize(new THREE.Vector3());
                                const center = box.getCenter(new THREE.Vector3());
                                selection.highlight.scale.set(size.x, size.y, size.z);
                                selection.highlight.position.copy(center);
                            }
                        } else {
                            selection.highlight.update();
                        }
                    }
                }
                if (orbitControls) orbitControls.update();
                if (renderer && scene && camera) renderer.render(scene, camera);
            };

            const clearModels = () => {
                mixers.forEach(m => m.stopAllAction());
                mixers = [];
                loadedModels.forEach(m => {
                    scene.remove(m.model);
                    if (m.hitProxy) scene.remove(m.hitProxy);
                });
                loadedModels = [];
                currentModel = null;
                deselectAll();
                statusLabel.innerText = "Ready";
            };

            const clearSkeletons = () => {
                for (const s of skeletalSamples) scene.remove(s.group);
                skeletalSamples = [];
                maxFrames = 0;
            };

            const clearScene = () => {
                console.log("[HY-Motion] Clearing entire scene...");
                clearModels();
                clearSkeletons();
                isPlaying = false;
                playBtn.innerText = "Play";
            };

            const centerCameraOnObject = (obj) => {
                if (!obj) return;
                setTimeout(() => {
                    const box = new THREE.Box3().setFromObject(obj);
                    if (box.isEmpty()) return;
                    const center = box.getCenter(new THREE.Vector3());
                    const size = box.getSize(new THREE.Vector3());
                    const maxDim = Math.max(size.x, size.y, size.z);
                    orbitControls.target.copy(center);
                    camera.position.set(center.x, center.y + (maxDim * 0.2), center.z + Math.max(maxDim * 2.5, 4));
                    orbitControls.update();
                }, 150);
            };

            const loadMotions = (motions) => {
                if (!Array.isArray(motions)) return;
                console.log("[HY-Motion] Loading Skeletons:", motions.length);
                clearSkeletons();
                const masterGroup = new THREE.Group();
                let allMaxFrames = 0;
                for (const m of motions) {
                    if (!m.keypoints || !m.keypoints[0]) continue;
                    const group = new THREE.Group();
                    group.position.x = (m.posX || 0) - (motions.length - 1) * 0.75;
                    const sampleColor = m.color || 0x3366ff;
                    const jointMat = new THREE.MeshStandardMaterial({ color: sampleColor, emissive: sampleColor, emissiveIntensity: 0.3 });
                    const boneMat = new THREE.LineBasicMaterial({ color: sampleColor, transparent: true, opacity: 0.8 });
                    const joints = [];
                    const firstFrame = m.keypoints[0];
                    const numJoints = firstFrame.length;
                    for (let j = 0; j < numJoints; j++) {
                        const size = j === 0 ? 0.12 : 0.05; // Slightly larger for easier clicking
                        const sph = new THREE.Mesh(new THREE.SphereGeometry(size, 12, 12), jointMat);
                        group.add(sph);
                        joints.push(sph);
                    }
                    const bones = [];
                    for (const [j1, j2] of SMPL_H_SKELETON) {
                        if (j1 < numJoints && j2 < numJoints) {
                            const geo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
                            const line = new THREE.Line(geo, boneMat);
                            group.add(line);
                            bones.push({ line, j1, j2 });
                        }
                    }
                    masterGroup.add(group);
                    skeletalSamples.push({ group, joints, bones, keypoints: m.keypoints });
                    allMaxFrames = Math.max(allMaxFrames, m.keypoints.length);
                }
                scene.add(masterGroup);
                maxFrames = allMaxFrames;
                currentFrame = 0;
                updateSkeletons(0);
                isPlaying = true;
                playBtn.innerText = "Pause";
                statusLabel.innerText = `${skeletalSamples.length} Skel${loadedModels.length ? ' + Model' : ''}`;
                // Only center if we don't have a model yet (model takes priority for centering)
                if (loadedModels.length === 0) centerCameraOnObject(masterGroup);
            };

            const updateSkeletons = (frameIdx) => {
                for (const sample of skeletalSamples) {
                    const f = frameIdx % sample.keypoints.length;
                    const pos = sample.keypoints[f];
                    if (!pos) continue;
                    for (let i = 0; i < sample.joints.length; i++) {
                        const p = pos[i];
                        if (p) sample.joints[i].position.set(p[0], p[1], p[2]);
                    }
                    for (const b of sample.bones) {
                        const p1 = pos[b.j1], p2 = pos[b.j2];
                        if (p1 && p2 && THREE) {
                            b.line.geometry.setFromPoints([new THREE.Vector3(p1[0], p1[1], p1[2]), new THREE.Vector3(p2[0], p2[1], p2[2])]);
                        }
                    }
                }
            };

            const loadGenericModel = async (modelPath, format, customName = null, index = 0, total = 1) => {
                if (!modelPath || modelPath === "none") return;

                // Ensure Three is loaded if called from widget before initThree finished
                if (!THREE && window.__HY_MOTION_THREE__) THREE = window.__HY_MOTION_THREE__;
                if (!THREE) {
                    console.log("[HY-Motion] Three not ready, waiting for init...");
                    await initThree();
                }

                console.log("[HY-Motion] Loading 3D Model Path:", modelPath, `(${index + 1}/${total})`);
                if (index === 0) {
                    clearModels();
                    statusLabel.innerText = "Loading Model...";
                }

                try {
                    let loaderModule = null;
                    let LoaderClass = null;

                    if (format === 'fbx') {
                        if (!window.__HY_MOTION_FBX_LOADER__) window.__HY_MOTION_FBX_LOADER__ = await import(FBX_LOADER_URL);
                        if (!window.fflate) window.fflate = await import("https://esm.sh/fflate@0.8.0");
                        LoaderClass = window.__HY_MOTION_FBX_LOADER__.FBXLoader;
                    } else if (format === 'glb' || format === 'gltf') {
                        if (!window.__HY_MOTION_GLTF_LOADER__) window.__HY_MOTION_GLTF_LOADER__ = await import(GLTF_LOADER_URL);
                        LoaderClass = window.__HY_MOTION_GLTF_LOADER__.GLTFLoader;
                    } else if (format === 'obj') {
                        if (!window.__HY_MOTION_OBJ_LOADER__) window.__HY_MOTION_OBJ_LOADER__ = await import(OBJ_LOADER_URL);
                        LoaderClass = window.__HY_MOTION_OBJ_LOADER__.OBJLoader;
                    }

                    if (!LoaderClass) throw new Error("Unsupported format: " + format);
                    const loader = new LoaderClass();
                    if (format === 'fbx' && window.fflate && typeof loader.setFflate === 'function') {
                        loader.setFflate(window.fflate);
                    }

                    // Parse folder and filename
                    let type = "output";
                    let filename = modelPath;
                    let subfolder = "";

                    if (modelPath.startsWith("input/")) {
                        type = "input";
                        filename = modelPath.substring(6);
                    } else if (modelPath.startsWith("output/")) {
                        type = "output";
                        filename = modelPath.substring(7);
                    }

                    if (filename.includes("/") || filename.includes("\\")) {
                        const parts = filename.split(/[/\\]/);
                        filename = parts.pop();
                        subfolder = parts.join("/");
                    }

                    let fetchUrl = `${window.location.origin}/view?type=${type}&filename=${encodeURIComponent(filename)}`;
                    if (subfolder) fetchUrl += `&subfolder=${encodeURIComponent(subfolder)}`;

                    const loadingLabel = customName || filename;
                    statusLabel.innerText = `Loading ${loadingLabel}...`;
                    statusLabel.style.color = "#ffaa00";

                    console.log("[HY-Motion] Loading 3D Model:", fetchUrl);
                    loader.load(fetchUrl, (result) => {
                        // Double check THREE is available in callback scope
                        const T = THREE || window.__HY_MOTION_THREE__;
                        if (!T) { console.error("[HY-Motion] THREE still null in callback!"); return; }

                        const fbx = (format === 'glb' || format === 'gltf') ? result.scene : result;
                        fbx.updateMatrixWorld(true);

                        const debugMat = new T.MeshStandardMaterial({ color: 0xffaa00, roughness: 0.5, metalness: 0.5, side: T.DoubleSide });
                        fbx.traverse((child) => {
                            if (child.isMesh) {
                                if (!child.material || child.material.type === "MeshBasicMaterial") child.material = debugMat;
                                child.castShadow = true;
                                child.receiveShadow = true;
                                child.frustumCulled = false;
                            }
                        });

                        scene.add(fbx);
                        currentModel = fbx;
                        loadedModels.push({
                            model: fbx,
                            name: customName || filename,
                            fileName: filename,
                            subfolder: subfolder,
                            fileType: type
                        });

                        // Positioning side-by-side
                        if (total > 1) {
                            fbx.position.x = index * 1.5 - (total - 1) * 0.75;
                        }

                        const box = new THREE.Box3().setFromObject(fbx);
                        const maxDim = Math.max(...box.getSize(new THREE.Vector3()).toArray());
                        if (maxDim > 0) fbx.scale.setScalar(1.7 / maxDim);

                        if (fbx.animations && fbx.animations.length > 0 || (result.animations && result.animations.length > 0)) {
                            const m = new THREE.AnimationMixer(fbx);
                            const anims = fbx.animations || result.animations;
                            m.clipAction(anims[0]).play();
                            mixers.push(m);
                            isPlaying = true;
                            playBtn.innerText = "Pause";
                        }

                        statusLabel.innerText = total > 1 ? `Loaded ${loadedModels.length}/${total}` : "Model Loaded";
                        if (total > 1) {
                            // Center on all models
                            const group = new THREE.Group();
                            loadedModels.forEach(m => group.add(m.model.clone())); // Dummy group for centering auto-box
                            centerCameraOnObject(group);
                        } else {
                            centerCameraOnObject(fbx);
                        }
                    }, (xhr) => {
                        if (xhr.lengthComputable) statusLabel.innerText = `Loading: ${Math.round((xhr.loaded / xhr.total) * 100)}%`;
                    }, (err) => {
                        console.error("[HY-Motion] Load fail:", err);
                        statusLabel.innerText = "Load Error";
                        statusLabel.style.color = "red";
                    });

                } catch (e) {
                    console.error("3D Load Error:", e);
                    statusLabel.innerText = "Error: " + e.message;
                    statusLabel.style.color = "red";
                }
            };

            const ensureString = (val) => {
                if (typeof val === 'string') return val;
                if (Array.isArray(val)) {
                    // Check if it's a character array (e.g. ['f', 'b', 'x'])
                    if (val.every(i => typeof i === 'string' && i.length <= 1)) return val.join('');
                    // Otherwise take the last element (ComfyUI pattern)
                    return ensureString(val[val.length - 1]);
                }
                return String(val);
            };

            const handleData = (data) => {
                if (!data) return;
                console.log("[HY-Motion] handleData received keys:", Object.keys(data));

                if (data.motions) {
                    try { loadMotions(typeof data.motions === 'string' ? JSON.parse(data.motions) : data.motions); }
                    catch (e) { console.error("[HY-Motion] JSON parse error:", e); }
                }

                // Handle both legacy 'fbx_url' and new 'fbx_paths'
                const rawUrl = data.fbx_url || data.fbx_paths;
                if (rawUrl) {
                    const urlStr = ensureString(rawUrl);
                    // Split by newline and load each model (usually just one, but supports multi-batch)
                    const urls = urlStr.split('\n').map(u => u.trim()).filter(u => u.length > 0);
                    urls.forEach((url, i) => {
                        console.log("[HY-Motion] Found FBX to load:", url);
                        // Extract filename for naming
                        const name = url.split(/[/\\]/).pop();
                        loadGenericModel(url, 'fbx', name, i, urls.length);
                    });
                }

                if (data.model_url) {
                    const url = ensureString(data.model_url);
                    const format = ensureString(data.format || 'fbx');
                    loadGenericModel(url, format);
                }
            };

            initThree();

            // Support live preview on widget change for the loader nodes
            if (nodeData.name === "HYMotion3DModelLoader" || nodeData.name === "HYMotionFBXPlayer") {
                const widgetName = nodeData.name === "HYMotion3DModelLoader" ? "model_path" : "fbx_name";
                setTimeout(() => {
                    const modelWidget = this.widgets?.find(w => w.name === widgetName);
                    if (modelWidget) {
                        const oldCallback = modelWidget.callback;
                        modelWidget.callback = function (value) {
                            if (oldCallback) oldCallback.apply(this, arguments);
                            console.log("[HY-Motion] Widget changed:", value);

                            let finalUrl = value;
                            if (nodeData.name === "HYMotionFBXPlayer" && value !== "none") {
                                // Add "output/" prefix for legacy player if not present
                                finalUrl = value.startsWith("output/") ? value : `output/${value}`;
                            }

                            const ext = finalUrl.split('.').pop().toLowerCase();
                            loadGenericModel(finalUrl, ext);
                        };
                        // Trigger initial load
                        if (modelWidget.value && modelWidget.value !== "none") {
                            let finalUrl = modelWidget.value;
                            if (nodeData.name === "HYMotionFBXPlayer") {
                                finalUrl = finalUrl.startsWith("output/") ? finalUrl : `output/${finalUrl}`;
                            }
                            const ext = finalUrl.split('.').pop().toLowerCase();
                            loadGenericModel(finalUrl, ext);
                        }
                    }
                }, 500);
            }

            node.onExecuted = function (output) {
                if (!isInitialized) pendingDataQueue.push(output);
                else handleData(output);
            };

            playBtn.onclick = (e) => { e.stopPropagation(); isPlaying = !isPlaying; playBtn.innerText = isPlaying ? "Pause" : "Play"; };
            cycleBtn.onclick = (e) => { e.stopPropagation(); cycleSelection(); };
            exportBtn.onclick = (e) => { e.stopPropagation(); exportSelected(); };
            progress.oninput = () => {
                isPlaying = false; playBtn.innerText = "Play";
                const p = progress.value / 100;
                if (mixers.length > 0) {
                    for (const m of mixers) {
                        if (m._actions[0]) m.setTime(p * m._actions[0]._clip.duration);
                    }
                }
                else if (maxFrames > 0) { currentFrame = Math.floor(p * maxFrames); updateSkeletons(currentFrame); }
            };
            return r;
        };
    }
});
