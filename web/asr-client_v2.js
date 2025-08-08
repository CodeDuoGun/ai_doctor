let recorder = null;
let socket = null;
let isRec = false;
let sampleBuf = new Int16Array();

const btnMic = document.getElementById("btn-mic");
const btnConnect = document.getElementById("ws-connect");
const chatInput = document.getElementById("chat-input");
const sendBtn = document.getElementById("btn-send");
const info_div = document.createElement("div");
document.body.appendChild(info_div);

// 初始化录音对象
const rec = Recorder({
  type: "pcm",
  bitRate: 16,
  sampleRate: 16000,
  onProcess: recProcess
});

// 合并 Int16Array
function concatInt16Array(a, b) {
  const c = new Int16Array(a.length + b.length);
  c.set(a, 0);
  c.set(b, a.length);
  return c;
}

// 录音回调
function recProcess(buffer, powerLevel, bufferDuration, bufferSampleRate) {
  if (!isRec) return;

  const data_48k = buffer[buffer.length - 1];
  const data_16k = Recorder.SampleData([data_48k], bufferSampleRate, 16000).data;
  sampleBuf = concatInt16Array(sampleBuf, data_16k);

  info_div.innerHTML = `${(bufferDuration / 1000).toFixed(2)}s 录音中...`;
}

// 添加消息到聊天框
function appendMessage(sender, text, type = "text") {
  const chatbox = document.getElementById("chat-box");
  const messageDiv = document.createElement("div");
  messageDiv.classList.add("message", sender === "user" ? "user" : "bot");

  const avatar = document.createElement("img");
  avatar.classList.add("avatar");
  avatar.src = sender === "user" ? "/static/user.png" : "/static/ai_girl.png";

  const bubble = document.createElement("div");
  bubble.classList.add("bubble");
  bubble.textContent = text;

  messageDiv.appendChild(avatar);
  messageDiv.appendChild(bubble);
  chatbox.appendChild(messageDiv);
  chatbox.scrollTop = chatbox.scrollHeight;
}

// 回车发送文字消息
// 公共发送函数
function sendMessage() {
    const text = chatInput.value.trim();
    if (!text) return;
    appendMessage("user", text);
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ msg: text }));
    } else {
      appendMessage("系统", "❌ WebSocket未连接");
    }
    chatInput.value = "";
  }
  
  // 回车发送
  chatInput.addEventListener("keydown", function (event) {
    if (event.isComposing) return;
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  });
  
  // 点击按钮发送
  sendBtn.addEventListener("click", function () {
    sendMessage();
  });

// 连接 WebSocket
btnConnect.addEventListener("click", () => {
  if (socket && socket.readyState === WebSocket.OPEN) {
    appendMessage("系统", "WebSocket 已连接");
    return;
  }
  socket = new WebSocket("ws://127.0.0.1:8765/");

  socket.onopen = async () => {
    appendMessage("系统", "✅ WebSocket连接成功");
    try {
      const resp = await fetch("http://127.0.0.1:8080/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ uid: "test" }),
      });
      const data = await resp.json();
      if (!resp.ok || data.status !== "ok") {
        appendMessage("错误", "❌ /start 接口调用失败");
        return;
      }
      appendMessage("系统", "✅ /start 接口调用成功");
      startWebRTC();
    } catch (err) {
      appendMessage("错误", "❌ /start 请求失败");
      console.error(err);
    }
  };

  socket.onmessage = (e) => {
    try {
      const msg = JSON.parse(e.data);
      if (msg.asr) appendMessage("user", msg.asr);
      if (msg.nlp) appendMessage("ai", msg.nlp);
    } catch (e) {
      console.error("解析失败:", e);
    }
  };

  socket.onerror = () => {
    appendMessage("错误", "WebSocket连接失败");
  };

  socket.onclose = () => {
    appendMessage("系统", "🔌 WebSocket已关闭");
    isRec = false;
  };
});

// 麦克风按钮：开始/结束录音
btnMic.addEventListener("click", () => {
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    alert("请先连接 WebSocket！");
    return;
  }

  if (!isRec) {
    // 开始录音
    rec.open(() => {
      rec.start();
      isRec = true;
      sampleBuf = new Int16Array();
      btnMic.textContent = "⏹ 停止";
      appendMessage("系统", "🎤 已开始录音");
    });
  } else {
    // 结束录音并发送
    const request = {
      chunk_size: [5, 10, 5],
      wav_name: "h5",
      is_speaking: false,
      chunk_interval: 10,
    };

    if (sampleBuf.length > 0) {
      socket.send(sampleBuf);
      sampleBuf = new Int16Array();
    }
    socket.send(JSON.stringify(request));

    // 🔁 录音结束，获取 PCM 数据并播放
    rec.stop(function (blob, duration) {
    console.log("PCM Blob:", blob);

    Recorder.pcm2wav(
        {
        sampleRate: 16000,
        bitRate: 16,
        blob: blob,
        },
        function (wavBlob, duration) {
        console.log("✅ WAV 生成成功，时长:", duration);

        const url = (window.URL || webkitURL).createObjectURL(wavBlob);
        const audio = document.getElementById("audio_record");
        audio.src = url;
        audio.controls = true;
        audio.play(); // 自动播放
        },
        function (errMsg) {
        console.error("❌ WAV 生成失败:", errMsg);
        }
    );
    }, function (errMsg) {
    console.error("❌ rec.stop 失败:", errMsg);
    });

    isRec = false;
    btnMic.textContent = "🎤";
    appendMessage("系统", "⏹ 录音结束，已发送数据,正在识别...");
  }
});
