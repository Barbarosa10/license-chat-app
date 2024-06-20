import React, { createContext, useContext, useState } from 'react';

const ConversationContext = createContext();

export const ConversationProvider = ({ children }) => {
  const [conversations, setConversations] = useState([]);

  const addConversation = (conversation) => {
    setConversations((prevConversations) => {
        const existingConversations = prevConversations || [];
        console.log(existingConversations);
    
        return [...existingConversations, conversation];
      });
  };
  
  const setConversationsAtInitialization = (conversations) => {
    setConversations(conversations);
  };

  const updateConversation = (updatedData) => {
    const getAllConversations = (prevConversations) => {
        const updatedConversations = [...prevConversations];
        const index = updatedConversations.findIndex((conversation) => conversation.id === updatedData.id);
    
        if (index !== -1) {
          updatedConversations[index] = { ...updatedConversations[index], ...updatedData };
          return updatedConversations;
        }
        else{
            addConversation(updatedData);
            console.log(conversations);
            
            return null;
        }
      };
      const allConversations = getAllConversations(conversations);

      if(allConversations != null){
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