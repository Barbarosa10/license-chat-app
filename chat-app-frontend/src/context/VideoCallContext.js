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
  const [callOn, setCall] = useState(false);
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
  const setCallOn= (callon) => {
    setCall(callon);
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
      console.log('Destroying connection:', connectionRef.current);
      try {
        connectionRef.current.destroy();
      } catch (error) {
          console.error('Error destroying connection:', error);
      } finally {
          connectionRef.current = null;
          console.log('Connection destroyed');
      }
    }
  };

  const closeCamera = async () => {
    if (myVideo && myVideo.current instanceof MediaStream) {
      try{
        await myVideo.current.getTracks().forEach(track => track.stop());
        myVideo.current = null;
        myVideo = null;
        setStream(null);
      }catch(error){
        console.log(error);
      }
    }

    if (userVideo && userVideo.current instanceof MediaStream) {
      try{
        await userVideo.current.getTracks().forEach(track => track.stop());
        userVideo.current = null;
        userVideo = null;
        setStream(null);
      }catch(error){
        console.log(error);
      }
    }

    // if (stream) {
    //   try{
    //     await stream.getTracks().forEach(track => track.stop());
    //     setStream(null);
    //   }catch(error){
    //     console.log(error);
    //   } finally {
    //     console.log(stream.getTracks())
    //   }
    // }
  }

  return (
    <VideoCallContext.Provider value={{ connectionRef, userVideo, myVideo, calling, receivingCall, callerSignal, stream, callAccepted, callEnded, caller, callOn, 
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
                                        setUsername: setCurrentUsername, 
                                        setCallOn: setCall
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