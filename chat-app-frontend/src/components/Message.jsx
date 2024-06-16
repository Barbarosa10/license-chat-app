import React, { useEffect, useRef } from 'react'
import { useChat } from "../context/ChatContext";
import { useUser } from "../context/UserContext";


const Message = ({message}) => {
    const { currentUser } = useUser();
    const { data } = useChat();

    const ref = useRef();

    useEffect(() => {
      ref.current?.scrollIntoView({ behavior: "smooth" });
    }, [message]);

    return(

        <div 
            ref={ref}
            className={`message ${message.sender === currentUser.username && "owner"}`}
        >
            <div className='messageInfo'>
                <img 
                    src={`data:image/;base64,
                        ${message.sender === currentUser.username
                            ? currentUser.avatarImage
                            : data.user.avatarImage}`
                        } 
                    alt="" 
                />
            </div>
            <div className='messageContent'>
                <p>{message.message}</p>
            </div>
        </div>        
    )
}

export default Message;