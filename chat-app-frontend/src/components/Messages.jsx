import React, { useContext, useEffect, useState } from 'react'
import Message from './Message'
import { receiveMessageRoute, host } from "../utils/ApiRoute";
import { useChat } from "../context/ChatContext";
import { useUser } from "../context/UserContext";
import { useMessages } from "../context/MessageContext";
import axios from "axios";


const Messages = ({socket}) => {
    // const [messages, setMessages] = useState([]);
    const {messages, setMessagesAtInitialization, addMessage} = useMessages();

    const { data } = useChat();

    useEffect(() => {   
        const fetchMessages = async() => {

            try{
                console.log(data.chatId);
                const response = await(axios.post(`${receiveMessageRoute}`, 
                    {"conversationId": data.chatId}
                ));
                // console.log(response.data);
                setMessagesAtInitialization(response.data);
                // setMessages(response.data);

            }catch(error){
              console.error('Error fetching messages:', error);
            }
        }

        fetchMessages();
    }, [data.chatId]);

    useEffect(() => {
        if (socket.current) {
            console.log("Socket is available")
            console.log(socket.current);
          socket.current.on("msg-receive", (msg) => {
            console.log(msg);
            addMessage(
                {
                    "message": msg.message,
                    "sender": msg.from,
                    "conversationId": msg.chatId,
                    "timestamp": Date.now()
                }
            )
          });
        }
      }, [socket.current]);

    return(
        <div className='messages'>
            {messages.map((m, index) => {
                return(<Message key={index} message={m}/>)
                
            })}
            {/* <Message/>
            <Message/>
            <Message/>
            <Message/>
            <Message/>
            <Message/>
            <Message/>
            <Message/>
            <Message/>
            <Message/>
            <Message/>
            <Message/>
            <Message/> */}
        
        </div>
    )
}

export default Messages