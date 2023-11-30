import React, { useContext, useEffect, useRef } from 'react'
import { useChat } from "../context/ChatContext";
import { useUser } from "../context/UserContext";


const Message = ({message}) => {
    const { currentUser } = useUser();
    const { data } = useChat();

    // console.log(message);
    // console.log("A0");

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
                <span>Just now</span>
            </div>
            <div className='messageContent'>
                <p>{message.message}</p>
                {/* {message.avatarImage && <img src={message.avatarImage} alt="" />} */}
                {/* <img src="https://images.pexels.com/photos/220453/pexels-photo-220453.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500" alt="" /> */}
            </div>
        </div>        
    )
}

export default Message;