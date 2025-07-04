// recorder-setup.js
let recorder = null;
let asrInterval = null;

document.getElementById("submit-button").addEventListener("click", () => {
  const text = document.getElementById("chat-input").value.trim();
  if (text) {
    appendMessage("用户", text);
    document.getElementById("chat-input").value = "";
    simulateBotResponse(text);
  }
});

document.getElementById("toggle-voice").addEventListener("click", () => {
  document.getElementById("chat-input").classList.add("hidden");
  document.getElementById("toggle-voice").classList.add("hidden");
  document.getElementById("toggle-text").classList.remove("hidden");

  startASRStream((text) => {
    appendMessage("用户语音", text);
    simulateBotResponse(text);
  });
});

document.getElementById("toggle-text").addEventListener("click", () => {
  stopASR();
  document.getElementById("chat-input").classList.remove("hidden");
  document.getElementById("toggle-voice").classList.remove("hidden");
  document.getElementById("toggle-text").classList.add("hidden");
});

document.getElementById("clear").addEventListener("click", () => {
  const chatbot = document.getElementById("chatbot");
  chatbot.innerHTML = '<div><strong>AI:</strong> 欢迎与我对话</div>';
  document.getElementById("chat-input").value = '';
});

document.querySelectorAll(".example").forEach(example => {
  example.addEventListener("click", () => {
    document.getElementById("chat-input").value = example.textContent;
  });
});

function recorderStart(onBlobReady) {
  Recorder.getPermission().then(() => {
    recorder = Recorder({
      type: "pcm",
      sampleRate: 16000,
      bitRate: 16
    });

    recorder.open(() => {
      recorder.start();
      asrInterval = setInterval(() => {
        recorder.stop((blob) => {
          onBlobReady(blob);
          recorder.start();
        });
      }, 500);
    });
  }).catch((err) => {
    alert("麦克风权限获取失败：" + err.message);
  });
}

function stopASR() {
  if (recorder) recorder.close();
  if (asrInterval) clearInterval(asrInterval);
  stopASRStream();
}
