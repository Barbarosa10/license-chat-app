import React from 'react'
import Messages from './Messages'
import Input from './Input'
import Videos from './Videos'
import Camera from '../images/camera.png'

const Chat = () => {
    return(
        <div className='chat'>
            <div className='chatInfo'>
                <span>Duciuc Danut</span>
                <div className="chatCallButton">
                    <img src={Camera} alt=""/>
                </div>
            </div>
            <Videos/>
            <Messages/>
            <Input/>
        </div>
    )
}

export default Chat