import React from 'react';
import { useNavigate } from "react-router-dom";
import {useUser} from "../context/UserContext";
import { logoutRoute } from "../utils/ApiRoute";
import { useVideoCall } from "../context/VideoCallContext";
import axios from '../utils/axiosConfig';

const Navbar = () => {
    const navigate = useNavigate();
    const { currentUser} = useUser();

    const { setStream, closeCamera, setCalling, setReceivingCall, destroyConnection } = useVideoCall();

    const logout = async () => {
        
        try{
            await axios.post(`${logoutRoute}/${currentUser._id}`);
            localStorage.clear(process.env.REACT_APP_LOCALHOST_KEY);
            localStorage.clear('token');

            closeCamera();
            setStream(null);
            setCalling(false);
            setReceivingCall(false);
            destroyConnection();
        }catch(error){
            console.error(error);
        }

        navigate("/login");
    }

    if (currentUser) {
        return(

            <div className='navbar'>
                <div className='logo'>
                    <span>Chat App</span>
                </div>

                <div className='user'>
                    <img src={`data:image/;base64,${currentUser.avatarImage}`} alt="" />
                    <span>{currentUser.username}</span>
                    <button onClick={logout}>Logout</button>
                </div>
            </div>
        )
    }
}

export default Navbar