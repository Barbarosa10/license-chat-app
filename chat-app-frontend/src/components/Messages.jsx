import React, { useContext, useEffect, useState } from 'react'
import Message from './Message'
import { receiveMessageRoute, host } from "../utils/ApiRoute";
import { useChat } from "../context/ChatContext";
import { useUser } from "../context/UserContext";
import { useMessages } from "../context/MessageContext";
import axios from "axios";
import {useConversation} from "../context/ConversationContext";


const Messages = ({socket}) => {
    // const [messages, setMessages] = useState([]);
    const {messages, setMessagesAtInitialization, addMessage} = useMessages();
    const { updateConversation } = useConversation();
    const { currentUser} = useUser();
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

    // useEffect(() => {
    //     if (socket.current) {
    //         console.log("Socket is available")

    //         const handleReceiveMessage = (msg) => {
    //             console.log(msg);
    //             addMessage({
    //               "message": msg.message,
    //               "sender": msg.from,
    //               "conversationId": msg.chatId,
    //               "timestamp": Date.now()
    //             });
    //             const updatedConversation = {
    //                 "_id": msg.chatId,
    //                 "participants": [msg.fromUsername, currentUser.username],
    //                 "lastMessage": msg.message,
    //                 "timestamp": Date.now(),
    //                 "__v": {
    //                     "$numberInt": "0"
    //                   }
    //             };
    //             updateConversation(updatedConversation);
    //         };
    //         // console.log(socket.current);
    //       socket.current.on("msg-receive", handleReceiveMessage);
    //       return () => {
    //         console.log("Component unmounted. Removing event listener.");
    //         socket.current.off("msg-receive", handleReceiveMessage);
    //       };
    //     }
    //   }, [socket.current, addMessage]);



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