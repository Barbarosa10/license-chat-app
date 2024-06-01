import React, { useEffect, useState, useRef }  from "react";
import {motion} from "framer-motion"
import { useNavigate, useLocation } from "react-router-dom";
import axios from "axios";
import {useUser} from "../context/UserContext";

import { allContactsRoute, allConversationsRoute, contactRoute, host } from "../utils/ApiRoute";
import { useChat } from "../context/ChatContext";
import { useVideoCall } from "../context/VideoCallContext";

import Sidebar from "../components/SideBar";
import Chat from "../components/Chat";
import Welcome from "../components/Welcome";
import CallPopup from "../components/CallPopup";
import { io } from "socket.io-client";
import {MessageProvider} from "../context/MessageContext";
import { useSocket } from "../context/SocketContext";

import Peer from "simple-peer";
import PopupResponse from "../components/PopupResponse";
import { usePopup } from '../context/PopupContext';


const localhost_key = "chat-app-current-user"

const Home = () => {
    const navigate = useNavigate();
    const { currentUser, setCurrentUser } = useUser();
    const { chatSelected, dispatch, selectChat } = useChat();
    const { userVideo, receivingCall, setStream, closeCamera, setMyVideo, stream, caller, callerSignal, setCallAccepted, setConnectionRef, setCalling, setReceivingCall, setCaller, setUsername, setCallerSignal, setUserVideo, destroyConnection } = useVideoCall();
    const { socketv, setSocket } = useSocket();
    const {showPopup, message} = usePopup();

    const socket = useRef();

    const [user, setUser] = useState(null);


    const [dummyState, setDummyState] = useState(0);
    const sk = useRef();

    const triggerRerender = () => {
      setDummyState((prev) => prev + 1);
    };

    useEffect(() => {
      socket.current = io(host);
      console.log(socket);

      socket.current.on("open-socket", (data) => {
        console.log(data);
        sk.current = io(host); 
        console.log(sk);
        setSocket(sk);

        sk.current.emit("add-user-video", data.to);
        console.log(socketv);


        sk.current.on("callUser", (data) => {
          console.log("calwaser");
          console.log(data);
          const buildUser = async() => {
            console.log("USER:")
            console.log(currentUser);
            if(currentUser){
              const avatarImage = await getAvatarImage(data.username);
              setUser({
                "id": data.chatId,
                "_id": data.from,
                "username": data.username,
                "participants": [data.username, currentUser.username],
                "lastMessage": data.lastMessage,
                "timestamp": data.timestamp,
                avatarImage: avatarImage
              });
              
              console.log(user);
  
              setCaller(data.from);
              setUsername(data.username);
              setCallerSignal(data.signal);
              console.log("AUUUUUUUUUUU");
              setReceivingCall(true);
            }
          }
          buildUser();
        });
        socket.current.emit("socket-opened", {chatId: data.chatId, to: data.from, from: data.to});
        
      });
      console.log(currentUser);
      if (currentUser) {
        socket.current.emit("add-user", currentUser._id);
        console.log(socket);
      }

    }, [currentUser]);

    useEffect(() => {


      const fetchData = () => {
        // console.log(currentUser);
        if (!localStorage.getItem(localhost_key)) {
          navigate("/login");
        }
        else {
          try {
            const userData = JSON.parse(localStorage.getItem(localhost_key));
            if(currentUser != userData){
              setCurrentUser(userData.user);
            // console.log(currentUser);

            }
          } catch (error) {
            console.error('Error parsing user data:', error);
          }
        }
      };
      fetchData();
    }, []);

    

    const getAvatarImage = async(username) => {
      try {
        const contact = await axios.get(`${contactRoute}/${username}`);
        if (contact.data[0]) {
          const avatarImage = contact.data[0].avatarImage;
          return avatarImage;
        }
      } catch (error) {
        console.error('Error fetching contact data:', error);
        return error;
      }
    }


    const answerCall =() =>  {
      triggerRerender();
      setCallAccepted(true);
      setCalling(true);
      navigator.mediaDevices.getUserMedia({ video: true, audio: false }).then((stream) => {
          console.log(stream);
          setStream(stream);
          setMyVideo(stream);

          console.log(socketv);
          const peer = new Peer({
            initiator: false,
            trickle: false,
            stream: stream
          });
          peer._debug = console.log;
          peer.on("signal", (data) => {
            socketv.current.emit("answerCall", { signal: data, to: caller });
          })
          peer.on("stream", (stream) => {
            console.log("uservideooooo    ");
            console.log(stream);
            setUserVideo(stream);
            triggerRerender();
          })

          console.log(peer);
          peer.signal(callerSignal);
          
          setConnectionRef(peer);

          socketv.current.on("callEnded", (data) => {
            console.log("disconnecteddddddd");
            console.log(data);
            closeCamera();
            setStream(null);
            setCalling(false);
            destroyConnection();
          });
    
          console.log(user);
          if(user){
            selectChat();
            dispatch({ type: "CHANGE_USER", user: user });
          }
          setReceivingCall(false);
      })

    }
  
    const declineCall = () => {
      setReceivingCall(false);
    }




    return(
        <motion.div initial={{x: -100, opacity: 0 }} animate={{x: 0, opacity: 1 }} transition={{ duration: 1}} className="home">
            <div className="container">
              <MessageProvider>
                {console.log(receivingCall)}
                {receivingCall ? (
                   <CallPopup onAnswer={answerCall} onDecline={declineCall} />
                ) : null}

                <Sidebar socket={socket}/>
                {!chatSelected ? (
                  <Welcome />
                ) : (
                  <Chat socket={socket}/>
                )}
              </MessageProvider>

            </div>
            {showPopup ? (
                   <PopupResponse message={message} />
                ) : null}
        </motion.div>
    )

}

export default Home