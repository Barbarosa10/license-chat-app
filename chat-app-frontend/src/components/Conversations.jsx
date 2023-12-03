import React, { useEffect, useState, useRef } from 'react'
import {useUser} from "../context/UserContext";
import {useConversation} from "../context/ConversationContext";
import { useNavigate, useLocation } from "react-router-dom";
import { useChat } from "../context/ChatContext";
import { useMessages } from "../context/MessageContext";
import axios from "axios";
import { allContactsRoute, allConversationsRoute, contactRoute, host } from "../utils/ApiRoute";

const Conversations = ({socket}) => {
    const navigate = useNavigate();
    const { currentUser} = useUser();
    const { conversations, setConversationsAtInitialization, updateConversation } = useConversation();
    const { data, dispatch, selectChat } = useChat();
    const {messages, setMessagesAtInitialization, addMessage} = useMessages();
    useEffect(() => {

        const fetchConversations = async() => {

            try{
              if(currentUser){
                const response = await(axios.get(`${allConversationsRoute}/${currentUser.username}`));
                // console.log(response);
                const conversations = await Promise.all(response.data.map(async (element) => {
                  const conversation = {};
                  conversation.lastMessage = element.lastMessage;
                  const participant = element.participants.find(username => username !== currentUser.username);
                  
                  if (participant) {
                    conversation.username = participant;
                    conversation.id = element._id;
                    // console.log(element);
                    conversation.timestamp = element.timestamp;
        
                  
                    try {
                      const contact = await axios.get(`${contactRoute}/${participant}`);
                      if (contact.data[0]) {
                        conversation.avatarImage = contact.data[0].avatarImage;
                        conversation._id = contact.data[0]._id;
                      }
                    } catch (error) {
                      console.error('Error fetching contact data:', error);
                    }
                    // console.log(conversation);
                    // addConversation(conversation);
                    return conversation;
                  }
                }));
                setConversationsAtInitialization(conversations);
              } else {
                  navigate("/login");
              }
            }catch(error){
              console.error('Error fetching conversations:', error);
            }
        }
      
        if(currentUser){
            fetchConversations();
        }
    }, [currentUser]);

    useEffect(() => {
        if (socket.current) {
            console.log("Socket is available")

            const handleReceiveMessage = (msg) => {
                console.log(msg);
                if(data.user._id === msg.from){
                    addMessage({
                        "message": msg.message,
                        "sender": msg.from,
                        "conversationId": msg.chatId,
                        "timestamp": Date.now()
                      });
                }
                
                const setConversation = async() => {
                    const updatedConversation = {
                        "id": msg.chatId,
                        "_id": msg.from,
                        "username": msg.fromUsername,
                        "participants": [msg.fromUsername, currentUser.username],
                        "lastMessage": msg.message,
                        "timestamp": Date.now()

                    };
                    try {
                        const contact = await axios.get(`${contactRoute}/${msg.fromUsername}`);
                        if (contact.data[0]) {
                          updatedConversation.avatarImage = contact.data[0].avatarImage;
                        }
                      } catch (error) {
                        console.error('Error fetching contact data:', error);
                      }
                    updateConversation(updatedConversation);
                }
                setConversation();

            };
            // console.log(socket.current);
          socket.current.on("msg-receive", handleReceiveMessage);
          return () => {
            console.log("Component unmounted. Removing event listener.");
            socket.current.off("msg-receive", handleReceiveMessage);
          };
        }
      }, [socket.current, addMessage]);

    const handleSelect = (u) => {
        selectChat()
        dispatch({ type: "CHANGE_USER", user: u });
    };

    return(

        <div className='conversations'>
            {conversations.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp)) 
            .map((conversation, index) => {
                
                return (
                    <div key={index} className='userConversation' onClick={() => handleSelect(conversation)}>
                        <img src={`data:image/;base64,${conversation.avatarImage}`} alt="" />
                        <div className='userConversationInfo'>
                            <span>{conversation.username}</span>
                            {/* {console.log(conversation.lastMessage)} */}
                            <p>{conversation.lastMessage}</p>
                        </div>
                    </div>
                )
            })}
        </div>
            //  {/* <div className='userConversation'>
            // //     <img src={`data:image/svg+xml;base64,${contact.avatarImage}`} alt="" />
            // //     <div className='userConversationInfo'>
            // //         <span>{contact.username}</span>
            // //         <p>Hello world!</p>
            // //     </div>
            // // </div>

        
    )
}

export default Conversations