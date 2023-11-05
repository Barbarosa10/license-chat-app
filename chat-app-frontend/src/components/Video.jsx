import React, { useEffect, useRef, useState } from "react"

// const [ stream, setStream ] = useState()
// const myVideo = useRef()

// function Videoo(){
//     useEffect(() => {
//         navigator.mediaDevices.getUserMedia({ video: true, audio: true }).then((stream) => {
//             setStream(stream)
//                 myVideo.current.srcObject = stream
//         })})
    
// }

const Video = () => {
    return(
        <div className='video'>
            {/* <video playsInline muted ref={myVideo} autoPlay style={{ width: "300px" }} /> */}
        </div>
    )
}

export default Video