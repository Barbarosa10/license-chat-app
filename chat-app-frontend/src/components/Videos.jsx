import React, { useEffect, useRef, useState } from "react"
import Video from './Video'
import EndCall from '../images/endCall.png'
import { useVideoCall } from '../context/VideoCallContext';
import { useChat } from "../context/ChatContext";
import { useSocket } from "../context/SocketContext";

const Videos = ({socket}) => {
    const { data } = useChat();
    const {callAccepted, callEnded, stream, setStream,  myVideo, userVideo, calling, setCallEnded, destroyConnection, closeCamera, setCalling} = useVideoCall();
    const myCurrentVideo = useRef();
    const userCurrentVideo = useRef();
    const { socketv, setSocket } = useSocket();

    const leaveCall = () => {
        // setCallEnded(true);
        // destroyConnection();
        closeCamera();
        setCallEnded(true);
        destroyConnection();
        setStream(null);
        setCalling(false);

        const data_to_send = {
            to: data.user._id,
            chatId: data.chatId
        }
        socketv.current.emit("end-call", data_to_send);
    }

    useEffect(() => {
        console.log(stream);
        if(stream)
            myCurrentVideo.current.srcObject = myVideo.current;
    }, [myVideo.current]);

    useEffect(() => {
        console.log(userVideo.current);
        if(callAccepted && !callEnded){
            console.log("AMENOOOOOOOOOOO")
            userCurrentVideo.current.srcObject = userVideo.current;
        }

    }, [userVideo.current]);

    return(
        <div className={`call ${calling === true}` }>
            <div className={`videos ${calling === true}` }>
                <div className='user'>
                {console.log("IIIII")}
                    {stream && <video  playsInline muted ref={myCurrentVideo} autoPlay style={{ width: "250px", height: "200px" }} /> }
                    {/* <Video/> */}
                </div>
                <div className='contact-user'>
                    {callAccepted && !callEnded ?
                        <video playsInline muted ref={userCurrentVideo} autoPlay style={{ width: "250px", height: "200px" }} /> :
                        null}
                    {/* <Video/> */}
                </div>
            </div>
            <div className='callBar'>
                <img src={EndCall} onClick={leaveCall}  alt=""/>
            </div>
        </div>
    )
}

export default Videos