import React, { useEffect, useState, useRef }  from "react";
import {motion} from "framer-motion"
import { useNavigate } from "react-router-dom";
import axios from '../utils/axiosConfig';
import {useUser} from "../context/UserContext";

import { contactRoute, host } from "../utils/ApiRoute";
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

const Home = () => {
    const navigate = useNavigate();
    const { currentUser, setCurrentUser } = useUser();
    const { chatSelected, dispatch, selectChat, disableChat } = useChat();
    const { receivingCall, setStream, closeCamera, setMyVideo, caller, callerSignal, setCallAccepted, setConnectionRef, setCalling, setReceivingCall, setCaller, setUsername, setCallerSignal, setUserVideo, destroyConnection, setCallOn } = useVideoCall();
    const { socketv, setSocket } = useSocket();
    const {showPopup, message} = usePopup();

    const socket = useRef();

    const [user, setUser] = useState(null);

    const [ dummyState, setDummyState ] = useState(0);
    const sk = useRef();

    const triggerRerender = () => {
      setDummyState((prev) => prev + 1);
    };

    useEffect(() => {
      socket.current = io(host);

      socket.current.on("open-socket", (data) => {
        sk.current = io(host); 
        setSocket(sk);

        sk.current.emit("add-user-video", data.to);

        sk.current.on("callUser", (data) => {
          const buildUser = async() => {
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
              setReceivingCall(true);
            }
          }
          buildUser();
        });
        socket.current.emit("socket-opened", {chatId: data.chatId, to: data.from, from: data.to});
        
      });
      if (currentUser) {
        socket.current.emit("add-user", currentUser._id);
      }

    }, [currentUser]);

    useEffect(() => {

      if (localStorage.getItem('token') == undefined) {
        navigate("/login");
      }
      const fetchData = () => {
        if (localStorage.getItem(process.env.REACT_APP_LOCALHOST_KEY) == undefined) {
          navigate("/login");
        }
        else {
          disableChat();
          try {
            const userData = JSON.parse(localStorage.getItem(process.env.REACT_APP_LOCALHOST_KEY));
            if(currentUser !== userData){
              setCurrentUser(userData);
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
      closeCamera();
      destroyConnection();
      setStream(null);
      triggerRerender();
      setCallAccepted(true);
      setCalling(true);
      navigator.mediaDevices.getUserMedia({ video: true, audio: false }).then((stream) => {
          setStream(stream);
          setMyVideo(stream);

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
            setCallOn(true);
            setUserVideo(stream);
            triggerRerender();
          })

          peer.on('close', () => {
            closeConnection();
          });

          peer.on('error', err => {
              console.error('Peer error:', err);
              closeConnection();
          });

          peer.signal(callerSignal);
          
          setConnectionRef(peer);

          socketv.current.on("callEnded", (data) => {
            closeConnection();
          });

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

    const closeConnection = () => {
      setCallOn(false);
      closeCamera();
      destroyConnection();
      setCalling(false);
      
  };

    return(
        <motion.div initial={{x: -100, opacity: 0 }} animate={{x: 0, opacity: 1 }} transition={{ duration: 1}} className="home">
            <div className="container">
              <MessageProvider>
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