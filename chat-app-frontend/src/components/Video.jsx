import React, { useEffect, useRef, useState } from "react"
import { useVideoCall } from '../context/VideoCallContext'
// const [ stream, setStream ] = useState()


// function Videoo(){
//     useEffect(() => {
//         navigator.mediaDevices.getUserMedia({ video: true, audio: true }).then((stream) => {
//             // setStream(stream)
//             myVideo.current.srcObject = stream
//         })})
    
// }

const Video = (video) => {
    const {calling, stream, myVideo, userVideo, setStream, setMyVideo, setUserVideo} = useVideoCall();
    const myCurrentVideo = useRef();
    // useEffect(() => {
    //     console.log(stream);
    //     myCurrentVideo.current.srcObject = stream;
    //     // navigator.mediaDevices.getUserMedia({ video: true, audio: true }).then((stream) => {
    //     //     // setStream(stream)
    //     //     myVideo.current.srcObject = stream;
    //     // })
    // }, []);
    useEffect(() => {
        console.log(stream);
        myCurrentVideo.current.srcObject = myVideo.current;
    }, [myVideo.current]);

    // useEffect(() => {
    //     console.log(stream);
    //     userCurrentVideo.current.srcObject = userVideo.current;
    // }, [userVideo.current]);

    return(
        <div className='video'>
            <video playsInline muted ref={myCurrentVideo} autoPlay style={{ width: "250px", height: "200px" }} />
        </div>
    )
}

export default Video