// WebRTC 视频接收逻辑
let pc = null;

function startWebRTC() {
  pc = new RTCPeerConnection({ sdpSemantics: 'unified-plan' });

  pc.addTransceiver('video', { direction: 'recvonly' });
  pc.addTransceiver('audio', { direction: 'recvonly' });

  pc.ontrack = (event) => {
    if (event.track.kind === 'video') {
      document.getElementById('video').srcObject = event.streams[0];
    } else if (event.track.kind === 'audio') {
      document.getElementById('audio').srcObject = event.streams[0];
      // // 创建 MediaRecorder 实例
      // mediaRecorder = new MediaRecorder(event.streams[0]);
      // recordedChunks = [];

      // mediaRecorder.ondataavailable = (event) => {
      //   if (event.data.size > 0) {
      //     recordedChunks.push(event.data);
      //   }
      // };

      // mediaRecorder.onstop = () => {
      //   const audioBlob = new Blob(recordedChunks, { type: 'audio/webm' });
      //   const audioURL = URL.createObjectURL(audioBlob);

      //   const audioPlayer = document.getElementById('audio_record');
      //   audioPlayer.src = audioURL;
      //   audioPlayer.load();
      // };

      // mediaRecorder.start();
      // console.log("🎙️ 已开始录音");
      }
  };

  pc.createOffer()
    .then(offer => pc.setLocalDescription(offer))
    .then(() => waitIceGatheringComplete())
    .then(() => {
      return fetch('/offer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sdp: pc.localDescription.sdp,
          type: pc.localDescription.type
        })
      });
    })
    .then(response => response.json())
    .then(answer => pc.setRemoteDescription(answer))
    .catch(e => alert('WebRTC error: ' + e));
}

function waitIceGatheringComplete() {
  return new Promise(resolve => {
    if (pc.iceGatheringState === 'complete') {
      resolve();
    } else {
      const check = () => {
        if (pc.iceGatheringState === 'complete') {
          pc.removeEventListener('icegatheringstatechange', check);
          resolve();
        }
      };
      pc.addEventListener('icegatheringstatechange', check);
    }
  });
}

