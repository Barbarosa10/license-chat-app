import React, {useState} from 'react'
import Messages from './Messages'
import Input from './Input'
import Videos from './Videos'
import Camera from '../images/camera.png'

import {useUser} from "../context/UserContext";
import { useChat } from "../context/ChatContext";
import { useVideoCall } from '../context/VideoCallContext';
import {MessageProvider} from "../context/MessageContext";

import Peer from "simple-peer";

const Chat = ({socket}) => {
    const { currentUser } = useUser();
    const { data } = useChat();
    const {calling, setCalling, myVideo, callAccepted, stream, setStream, setMyVideo, setUserVideo, setCallAccepted, setConnectionRef} = useVideoCall();
    // console.log(data?.user);
    const [dummyState, setDummyState] = useState(0);

    // Manually trigger a re-render
    const triggerRerender = () => {
      setDummyState((prev) => prev + 1);
    };

	const callUser = () => {
        setCalling(true);
         navigator.mediaDevices.getUserMedia({ video: true, audio: true }).then((stream) => {
            console.log(stream);
            setStream(stream);
            setMyVideo(stream);
            // myVideo.current.srcObject = stream;
            console.log("YAAA");
            console.log(stream);
            const peer = new Peer({
                initiator: true,
                trickle: false,
                stream: stream
            });
            peer.on("signal", (signalData) => {
                socket.current.emit("callUser", {
                    userToCall: data.user._id,
                    chatId: data.chatId,
                    lastMessage: data.user.lastMessage,
                    timestamp: data.user.timestamp,
                    signalData: signalData,
                    from: currentUser._id,
                    username: currentUser.username
                });
            });
            peer.on("stream", (stream) => { 
                setUserVideo(stream);
                triggerRerender();
            });
            socket.current.on("callAccepted", (signal) => {
                setCallAccepted(true);
                peer.signal(signal);
            });
    
            setConnectionRef(peer);
        })


	}

    return(
        <div className='chat'>
            <div className='chatInfo'>
                {console.log(data.user)}
                <span>{data.user?.username}</span>
                <div className="chatCallButton">
                    <img src={Camera} onClick={callUser} alt=""/>
                </div>
            </div>

            {/* <MessageProvider> */}
                <Videos/>
                <Messages socket={socket}/> 
                <Input socket={socket}/>
            {/* </MessageProvider> */}
        </div>
    )
}

export default Chat