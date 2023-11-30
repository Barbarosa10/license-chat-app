import React from 'react'
import Messages from './Messages'
import Input from './Input'
import Videos from './Videos'
import Camera from '../images/camera.png'

import { useChat } from "../context/ChatContext";
import {MessageProvider} from "../context/MessageContext";

const Chat = ({socket}) => {
    const { data } = useChat();
    // console.log(data?.user);

    return(
        <div className='chat'>
            <div className='chatInfo'>
                <span>{data.user?.username}</span>
                <div className="chatCallButton">
                    <img src={Camera} alt=""/>
                </div>
            </div>
            {/* <Videos/> */}
            {/* <MessageProvider> */}
                <Messages socket={socket}/> 
                <Input socket={socket}/>
            {/* </MessageProvider> */}
        </div>
    )
}

export default Chat