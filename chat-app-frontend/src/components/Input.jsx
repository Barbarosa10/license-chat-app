import React, { useContext, useState, useRef, useEffect } from 'react'
import { host, sendMessageRoute, updateConversationRoute } from "../utils/ApiRoute";
import { useMessages } from "../context/MessageContext";
import {useConversation} from "../context/ConversationContext";
import {useUser} from "../context/UserContext";
import { useChat } from "../context/ChatContext";
import axios from "axios";


const Input = ({socket}) => {
    const [text, setText] = useState("");
    const { currentUser, setCurrentUser } = useUser();
    const { data } = useChat();
    const {messages, addMessage} = useMessages();
    const { conversations, setConversationsAtInitialization } = useConversation();



    // useEffect(() => {
    //     if (currentUser) {
    //       socket.current = io(host);
    //     }
    //   }, [currentUser]);

    const handleSend = async () => {
        socket.current.emit("send-msg", {
            to: data.user._id,
            from: currentUser._id,
            message: text,
            chatId: data.chatId
        });

        const message = await axios.post(sendMessageRoute, {
            conversationId: data.chatId,
            sender: currentUser.username,
            message: text,

        });
        addMessage(message.data);

        const conversation = await axios.post(updateConversationRoute, {
            conversationId: data.chatId,
            lastMessage: text
        })
        // console.log(conversation.data);
        setText("");
    }

    return(
        <div className='input'>
            <input type="text"
            placeholder='Type something...'
            onChange={(e) => setText(e.target.value)}
            value={text}
            />
            <div className='send'>
                <button onClick={handleSend}>Send</button>
            </div>
        </div>
    )
}

export default Input