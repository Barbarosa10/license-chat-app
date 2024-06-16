import React, { useEffect, useState } from 'react'
import Message from './Message'
import { receiveMessageRoute } from "../utils/ApiRoute";
import { useChat } from "../context/ChatContext";
import { useMessages } from "../context/MessageContext";
import axios from '../utils/axiosConfig';
import { useVideoCall } from '../context/VideoCallContext';


const Messages = () => {
    const {calling} = useVideoCall();
    const {messages, setMessagesAtInitialization} = useMessages();
    const { data } = useChat();
    useEffect(() => {   
        const fetchMessages = async() => {

            try{
                const response = await(axios.post(`${receiveMessageRoute}`, 
                    {"conversationId": data.chatId}
                ));
                setMessagesAtInitialization(response.data);

            }catch(error){
              console.error('Error fetching messages:', error);
            }
        }

        fetchMessages();
    }, [data.chatId]);

    return(

        calling === false ? (
            <div className='messages' style={{ height: 'calc(100% - 130px)' }} >
              {messages.map((m, index) => {
                return <Message key={index} message={m} />;
              })}
            </div>
          ) : (
            <div className='messages'>
              {messages.map((m, index) => {
                return <Message key={index} message={m} />;
              })}
            </div>
          )
    )
}

export default Messages