import React, {useState, useRef} from 'react'
import Messages from './Messages'
import Input from './Input'
import Videos from './Videos'
import Camera from '../images/camera.png'

import {useUser} from "../context/UserContext";
import { useChat } from "../context/ChatContext";
import { useVideoCall } from '../context/VideoCallContext';
import { host } from "../utils/ApiRoute";
import { io } from "socket.io-client";
import { useSocket } from "../context/SocketContext";

import Peer from "simple-peer";

const Chat = ({socket}) => {
    const { currentUser } = useUser();
    const { data } = useChat();
    const { setCalling, stream, setStream, setMyVideo, setUserVideo, setCallAccepted, setConnectionRef, destroyConnection, closeCamera, setCallEnded} = useVideoCall();
    const [ dummyState, setDummyState] = useState(0);
    const { setSocket } = useSocket();
    const sk = useRef();

    const triggerRerender = () => {
      setDummyState((prev) => prev + 1);
    };
    const requestToOpenSocket = () => {
        const data_to_send = {
            to: data.user._id,
            from: currentUser._id,
            chatId: data.chatId
        }
        socket.current.emit("open-socket", data_to_send);

        socket.current.on("socket-opened", (data) => {
            callUser();
        });
    }
	const callUser = () => {
        triggerRerender();
        setCalling(true);
         navigator.mediaDevices.getUserMedia({ video: true, audio: true }).then((stream) => {
            setStream(stream);
            setMyVideo(stream);

            sk.current = io(host);
            setSocket(sk);
            sk.current.emit("add-user-video", currentUser._id);
            
            const peer = new Peer({
                initiator: true,
                trickle: false,

                stream: stream
            });
            peer._debug = console.log;
            peer.on("signal", (signalData) => {
                sk.current.emit("callUser", {
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
            peer.on("close", () => {
                closeConnection();
            });
            peer.on("error", (err) => {
                console.error("Peer connection error: ", err);
                closeConnection();
            });

            sk.current.on("callAccepted", (signal) => {
                setCallAccepted(true);
                setCallEnded(false);
                peer.signal(signal);
                
            });
            sk.current.on("callEnded", (data) => {
                closeConnection();

              });

            setConnectionRef(peer);
        });

        const closeConnection = () => {
            destroyConnection();
            closeCamera();
            setCalling(false);
        }

	}

    return(
        <div className='chat'>
            <div className='chatInfo'>
                <span>{data.user?.username}</span>
                <div className="chatCallButton">
                    <img src={Camera} onClick={requestToOpenSocket} alt=""/>
                </div>
            </div>

            {stream && <Videos/>}
            <Messages/> 
            <Input socket={socket}/>
        </div>
    )
}

export default Chat