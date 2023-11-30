import React, { createContext, useContext, useState } from 'react';
import Conversations from '../components/Conversations';

const ConversationContext = createContext();

export const ConversationProvider = ({ children }) => {
  const [conversations, setConversations] = useState([]);

  const addConversation = (conversation) => {
    setConversations((prevConversations) => {
        // Use an empty array if prevConversations is falsy
        const existingConversations = prevConversations || [];
        console.log(existingConversations);
    
        // Create a new array with the existing conversations and the new one
        return [...existingConversations, conversation];
      });
  };
  
  const setConversationsAtInitialization = (conversations) => {
    setConversations(conversations);
  };

  const updateConversation = (updatedData) => {
    // let index = -1;
    const getAllConversations = (prevConversations) => {
        const updatedConversations = [...prevConversations];
        const index = updatedConversations.findIndex((conversation) => conversation.id === updatedData.id);
    
        if (index !== -1) {
          updatedConversations[index] = { ...updatedConversations[index], ...updatedData };
          console.log("index1: " + index);
          return updatedConversations;
        }
        else{
            console.log("index: " + index);
            addConversation(updatedData);
            console.log(conversations);
            console.log("YASS");
            
            return null;
        }
      };
      const allConversations = getAllConversations(conversations);

      if(allConversations != null){
        console.log("NULLL");
        console.log(allConversations);
        setConversations(allConversations);
      }

  };

  return (
    <ConversationContext.Provider value={{ conversations, addConversation, setConversationsAtInitialization, updateConversation }}>
      {children}
    </ConversationContext.Provider>
  );
};

export const useConversation = () => {
  const context = useContext(ConversationContext);
  if (!context) {
    throw new Error('useConversation must be used within a ConversationProvider');
  }
  return context;
};