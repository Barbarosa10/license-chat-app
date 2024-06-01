import React, { createContext, useContext, useState, useRef } from 'react';

const VideoCallContext = createContext();

export const VideoCallProvider = ({ children }) => {
	const [ me, setMe ] = useState("");
	let [ stream, setStream ] = useState();
	const [ receivingCall, setReceivingCall ] = useState(false);
	const [ caller, setCaller ] = useState("");
	const [ callerSignal, setCallerSignal ] = useState();
	const [ callAccepted, setCallAccepted ] = useState(false);
	const [ idToCall, setIdToCall ] = useState("");
	const [ callEnded, setCallEnded] = useState(false);
	const [ username, setUsername ] = useState("");
  const [ calling, setCalling ] = useState(false);
  const myVideo = useRef();
	const userVideo = useRef();
	const connectionRef= useRef();


  const setCurrentMe = (me) => {
    setMe(me);
  };
  const setCurrentStream = (stream) => {
    setStream(stream);
  };
  const setCurrentReceivingCall = (receivingCall) => {
    setReceivingCall(receivingCall);
  };
  const setCurrentCaller = (caller) => {
    setCaller(caller);
  };
  const setCurrentCallerSignal = (callerSignal) => {
    setCallerSignal(callerSignal);
  };
  const setCurrentCallAccepted = (callAccepted) => {
    setCallAccepted(callAccepted);
  };
  const setCurrentIdToCall = (idToCall) => {
    setIdToCall(idToCall);
  };
  const setCurrentCallEnded = (callEnded) => {
    setCallEnded(callEnded);
  };
  const setCurrentUsername = (username) => {
    setUsername(username);
  };
  const setCurrentCalling = (calling) => {
    setCalling(calling);
  };

  const setCurrentMyVideo = (stream) => {
    myVideo.current = stream;
  };

  const setCurrentUserVideo = (stream) => {
    userVideo.current = stream;
  };

  const setCurrentConnectionRef = (connection) => {
    connectionRef.current = connection;
  };
  const destroyConnection = () => {
    if(connectionRef && connectionRef.current){
      connectionRef.current.destroy();
      connectionRef.current = null;
    }

  };

  const closeCamera = () => {
    console.log("STREAMMM: ");
    console.log(stream);

    if (stream) {
      console.log(stream.getTracks())
      console.log("STOP TRACKS")
      // Stop all tracks in the media stream
      try{
        stream.getTracks().forEach(track => track.stop());
        // stream.getTracks().forEach(track => console.log(track));
      }catch(error){
        console.log(error);
      }
      console.log(stream.getTracks())
      console.log(stream);
      // Optionally, clear the reference to the media stream
      // stream = null;
    }
  }

  return (
    <VideoCallContext.Provider value={{ connectionRef, userVideo, myVideo, calling, receivingCall, callerSignal, stream, callAccepted, callEnded, caller, 
                                        closeCamera,
                                        destroyConnection,
                                        setCalling: setCurrentCalling,
                                        setCaller: setCurrentCaller,
                                        setConnectionRef: setCurrentConnectionRef,
                                        setMyVideo: setCurrentMyVideo,
                                        setUserVideo: setCurrentUserVideo,
                                        setMe: setCurrentMe, 
                                        setStream: setCurrentStream,
                                        setReceivingCall: setCurrentReceivingCall,
                                        setCallerSignal: setCurrentCallerSignal,
                                        setCallAccepted: setCurrentCallAccepted,
                                        setIdToCall: setCurrentIdToCall,
                                        setCallEnded: setCurrentCallEnded,
                                        setUsername: setCurrentUsername
                                    }}>
      {children}
    </VideoCallContext.Provider>
  );
};

export const useVideoCall = () => {
  const context = useContext(VideoCallContext);
  if (!context) {
    throw new Error('useUser must be used within a UserProvider');
  }
  return context;
};