import React, { useEffect, useState, useRef }  from "react";
import {motion} from "framer-motion"
import { useNavigate, useLocation } from "react-router-dom";
import axios from "axios";
import { allContactsRoute, allConversationsRoute, contactRoute, host } from "../utils/ApiRoute";

import Sidebar from "../components/SideBar";
import Chat from "../components/Chat";

const localhost_key = "chat-app-current-user"

const Home = () => {
    const navigate = useNavigate();
    const [currentUser, setCurrentUser] = useState(undefined);
    const [contacts, setContacts] = useState([]);
    const [conversations, setConversations] = useState([]);

    useEffect(() => {
        const fetchData = () => {
          if (!localStorage.getItem(localhost_key)) {
            navigate("/login");
          } else {
            try {
              const userData = JSON.parse(localStorage.getItem(localhost_key));
              if(currentUser != userData.user){
                setCurrentUser(userData);
              console.log(currentUser);
              }
            } catch (error) {
              console.error('Error parsing user data:', error);
            }
          }
        };
        
        fetchData();
      }, []); // Include dependencies as needed
      
    const fetchContact = async(username) => {
      try{
        if(currentUser){
          const response = await(axios.get(`${contactRoute}/${username}`));
          // console.log(response.data);
          return response.data;
        } else {
            return null;
        }
      }catch(error){
        console.error('Error fetching conversations:', error);
      }
    }
    
    useEffect(() => {
    //   const fetchContacts = async () => {
    //       try {
    //           if (currentUser) {
    //               const data = await axios.get(`${allContactsRoute}/${currentUser.user._id}`);
    //               console.log(data.data);
    //               setContacts(data.data);
    //           } else {
    //               navigate("/login");
    //           }
    //       } catch (error) {
    //           console.error('Error fetching contacts:', error);
    //       }
    //   };
    //   if(currentUser){
    //       fetchContacts();
    //   }

    const fetchConversations = async() => {
      let conversations = [];
      try{
        if(currentUser){
          const response = await(axios.get(`${allConversationsRoute}/${currentUser.user.username}`));
          console.log(response.data);
          // response.data.forEach((element, index) => {
          //   let conversation = {};

          //   element.participants.forEach(async (username, index) => {
          //     if(username != currentUser.user.username){
          //       conversation.username = username;
          //       conversation.id = element._id;
          //       // try{
          //       //   const contact = await(axios.get(`${contactRoute}/${username}`));
          //       //   if(contact){
          //       //     conversation.avatarImage = contact.data[0].avatarImage;
          //       //   }
          //       // }catch (error) {
          //       //   console.error('Error fetching contact data:', error);
          //       // }
          //       conversations.push(conversation);
          //     }
          //   });
          // });
          const conversations = await Promise.all(response.data.map(async (element) => {
            const conversation = {};
            conversation.message = element.lastMessage;
            const participant = element.participants.find(username => username !== currentUser.user.username);
            
            if (participant) {
              conversation.username = participant;
              conversation.id = element._id;
  
            
              try {
                const contact = await axios.get(`${contactRoute}/${participant}`);
                if (contact.data[0]) {
                  conversation.avatarImage = contact.data[0].avatarImage;
                }
              } catch (error) {
                console.error('Error fetching contact data:', error);
              }
            
              return conversation;
            }
          }));
          setConversations(conversations);

          // response.data.forEach((element, index) => {
          //   let conversation = {};

          //   element.participants.forEach(async (username, index) => {
          //     if(username != currentUser.user.username){
          //       conversation.username = username;
          //       conversation.id = element._id;

          //       try{
          //         const contact = await(axios.get(`${contactRoute}/${username}`));
          //         if(contact){
          //           conversation.avatarImage = contact.data[0].avatarImage;
          //         }
          //       }catch (error) {
          //         console.error('Error fetching contact data:', error);
          //       }

          //       conversations.push(conversation);
          //     }

          //   })
          //   // console.log(conversations);
          //   console.log("\naaaaaaaaaaaa\n");
          // });
          // setConversations(response.data);
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

    return(
        <motion.div initial={{x: -100, opacity: 0 }} animate={{x: 0, opacity: 1 }} transition={{ duration: 1}} className="home">
            <div className="container">

                <Sidebar conversations={conversations}/>
                <Chat/>
            </div>
        </motion.div>
    )

}

export default Home