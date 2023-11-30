import React from 'react';
import { useNavigate, Link, useLoaderData } from "react-router-dom";
import {useUser} from "../context/UserContext";
import profileImage from "../images/addAvatar.png";

const localhost_key = "chat-app-current-user"

const Navbar = () => {
    const navigate = useNavigate();
    const { currentUser} = useUser();

    const logout = () => {
        localStorage.clear(localhost_key);
        navigate("/login");
    }
    // const userData = JSON.parse(localStorage.getItem(localhost_key));
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
    // else{
    //     navigate("/login");
    // }
}

export default Navbar