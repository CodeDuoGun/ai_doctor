let recorder = null;
let socket = null;
let isRec = false;

// 录音; 定义录音对象,wav格式
var rec = Recorder({
	type:"pcm",
	bitRate:16,
	sampleRate:16000,
	onProcess:recProcess
});

var info_div = document.getElementById('info_div');
var sampleBuf=new Int16Array();
// 定义按钮响应事件
var btnStart = document.getElementById('start-asr');
// btnStart.onclick = record;
var btnStop = document.getElementById('stop-asr');
// btnStop.onclick = stop;
btnStop.disabled = true;
btnStart.disabled = true;
 
btnConnect= document.getElementById('ws-connect');
// btnConnect.onclick = start;

// 连接按钮：建立 WebSocket + 启动 WebRTC
btnConnect.addEventListener("click", () => {
  if (socket && socket.readyState === WebSocket.OPEN) {
    appendMessage("系统", "WebSocket 已连接");
    return;
  }

  const wsURL = "ws://127.0.0.1:8765/"; // 请替换成你的 FunASR 地址
  socket = new WebSocket(wsURL);

  socket.onopen = async () => {
    appendMessage("系统", "✅ WebSocket连接成功");
    try {
      // ✅ 连接成功后，调用 /start 接口
      const response = await fetch("http://127.0.0.1:8080/start", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ uid: "test" }),
      });

      const respJson = await response.json();
      // appendMessage("系统", "chat json: ", respJson)

      if (!response.ok || respJson.status !== "ok") {
        appendMessage("错误", "❌ /start 接口调用失败");
        return;
      }

      appendMessage("系统", "✅ /start 接口调用成功");

      // ✅ 然后再启动 WebRTC
      startWebRTC();
      isRec = true;
      btnStart.disabled = false;
      btnStop.disabled = false;
      btnConnect.disabled = false;

    } catch (err) {
      appendMessage("错误", "❌ /start 请求失败");
      console.error(err);
    }
  };

  socket.onmessage = (e) => {
    try {
      const msg = JSON.parse(e.data);
      console.log(msg)
      if (msg.asr) {
        appendMessage("识别结果", msg.asr);
      }
      if (msg.nlp) {
        appendMessage("数字人结果", msg.nlp);
      }
    } catch (e) {
      console.error("解析失败:", e);
    }
  };

  socket.onerror = (err) => {
    appendMessage("错误", "WebSocket连接失败");
    console.error(err);
  };

  socket.onclose = () => {
    appendMessage("系统", "🔌 WebSocket已关闭");
    btnStart.disabled = true;
    btnStop.disabled = true;
    btnConnect.disabled = false;
  };
});

// 点击“开始识别”：启动录音
btnStart.addEventListener("click", () => {
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    alert("请先连接 WebSocket！");
    return;
  }

   
    rec.open( function(){
    rec.start();
    console.log("开始");
    appendMessage("系统", "🎤 已开始录音");
        btnStart.disabled = false;
        btnStop.disabled = false;
        btnConnect.disabled=true;
    });

});

// 点击“结束识别”：停止录音，发送数据给后端
btnStop.addEventListener("click", () => {
  var chunk_size = new Array(5, 10, 5);
  var request = {
    "chunk_size": chunk_size,
    "wav_name": "h5",
    "is_speaking": false,
    "chunk_interval": 10,
  };
  console.log(request);

  if (sampleBuf.length > 0) {
    socket.send(sampleBuf);
    console.log("sampleBuf.length: " + sampleBuf.length);
    sampleBuf = new Int16Array();
  }

  socket.send(JSON.stringify(request));
  isRec = false;
  info_div.innerHTML = "发送完数据,请等候,正在识别...";

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

//    if(isfilemode==false){
//         btnStop.disabled = true;
//         btnStart.disabled = true;
//         btnConnect.disabled=true;
//     //wait 3s for asr result
//     setTimeout(function(){
//         console.log("call stop ws!");
//         socket.close();
//         btnConnect.disabled=false;
//         info_div.innerHTML="请点击连接";}, 3000 );
    
//     rec.stop(function(blob,duration){
//         console.log(blob);
//         var audioBlob = Recorder.pcm2wav(data = {sampleRate:16000, bitRate:16, blob:blob},
//         function(theblob,duration){
//                 console.log(theblob);
//         var audio_record = document.getElementById('audio_record');
//         audio_record.src =  (window.URL||webkitURL).createObjectURL(theblob); 
//         audio_record.controls=true;

//         }, function(msg){console.log(msg)});
    

//         },function(errMsg){
//             console.log("errMsg: " + errMsg);
//         }
//     );
    
//     }
})

function concatInt16Array(a, b) {
  var c = new Int16Array(a.length + b.length);
  c.set(a, 0);
  c.set(b, a.length);
  return c;
}
// 录音回调处理函数
function recProcess(buffer, powerLevel, bufferDuration, bufferSampleRate, newBufferIdx, asyncEnd) {
  if (!isRec) return;

  // 取最后一帧数据（48kHz）
  var data_48k = buffer[buffer.length - 1];
  var array_48k = new Array(data_48k);

  // 采样率转换：48kHz -> 16kHz
  var data_16k = Recorder.SampleData(array_48k, bufferSampleRate, 16000).data;
  sampleBuf = concatInt16Array(sampleBuf, data_16k);

  // 合并到 sampleBuf
  // sampleBuf = Int16Array.from([...sampleBuf, ...data_16k]);

  info_div.innerHTML = `${(bufferDuration / 1000).toFixed(2)}s 录音中...`;

  // 每960个样本（约60ms）发送一次小块数据（你可以选择是否实时发送）
  // while (sampleBuf.length >= 960) {
  //   let sendBuf = sampleBuf.slice(0, 960);
  //   sampleBuf = sampleBuf.slice(960);

  //   if (socket && socket.readyState === WebSocket.OPEN) {
  //     console.log("发送音频数据中")
  //     // socket.send(sendBuf.buffer);
  //   }
  // }
}
