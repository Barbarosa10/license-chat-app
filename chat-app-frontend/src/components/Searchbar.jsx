import React, {useContext, useState} from 'react'
import axios from "axios";
import { allConversationsRoute, contactRoute, createConversationRoute } from "../utils/ApiRoute";
import Conversations from './Conversations';
import {useUser} from "../context/UserContext";
import {useConversation} from "../context/ConversationContext";

const localhost_key = "chat-app-current-user"

const Searchbar = () => {
    const [username, setUsername] = useState("");
    const [user, setUser] = useState(null);
    const [err, setErr] = useState(false);
    const { currentUser} = useUser();
    const { conversations, addConversation } = useConversation();
    // let currentUser = null;

    //get Current User
    // const currentUserData = JSON.parse(localStorage.getItem(localhost_key));
    // if(currentUserData)
    //     currentUser = currentUserData.user;



    const handleSearch = async () => {
        const fetchContact = async () => {
            try {
                const response = await axios.get(`${contactRoute}/${username}`);
                console.log(response.data);
                if(response.data[0].length == 0)
                    setErr(true);
                else
                setErr(false);
                setUser(response.data[0]);


            } catch (error) {
                console.error('Error fetching contact:', error);
                setErr(true);
            }
        };

        fetchContact();
    }

    const handleKeyDown = (e) => {
        e.code === "Enter" && handleSearch();
    }

    const handleSelect = async () => {
        let conversationExists = false;
        try{
            if(user.username == currentUser.username)
                throw("Can t chat with yourself");
            const response = await(axios.get(`${allConversationsRoute}/${currentUser.username}`));
            outerLoop: for (const element of response.data) {
                for (const username of element.participants) {
                    if (username !== currentUser.username && username === user.username) {
                    conversationExists = true;
                    break outerLoop;
                    }
                }
            }
            console.log(conversationExists);
            if(!conversationExists){
                const response = await axios.post(createConversationRoute, {
                    "participants": [currentUser.username, user.username]
                });
            
                const conversation = {};
                conversation.message = response.data.lastMessage;
                conversation.username = response.data.participants.find(username => username !== currentUser.username);
                conversation.id = response.data._id;
                conversation.timestamp = response.data.timestamp;
                try {
                    const contact = await axios.get(`${contactRoute}/${conversation.username}`);
                    if (contact.data[0]) {
                      conversation.avatarImage = contact.data[0].avatarImage;
                      conversation._id = contact.data[0]._id;
                    }
                  } catch (error) {
                    console.error('Error fetching contact data:', error);
                }
                addConversation(conversation);
                // console.log(response.data);
            }
            else{
                console.log("Already exists!")
            }
        }catch(error){console.log(error);}
        setUser(null);
        setUsername("");
    }

    return(
        <div className="search">
            <div className='searchForm'>
                <input 
                type="text" 
                placeholder='Search for an user'
                onKeyDown={handleKeyDown}
                onChange={(e) => setUsername(e.target.value)}
                value={username}/>
            </div>
            {err && <span>User not found!</span>}
            {!err && user && (
                <div className='userConversation' onClick={handleSelect}>
                    {/* <img src="https://images.pexels.com/photos/220453/pexels-photo-220453.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500" alt="" /> */}
                    <img src={`data:image/;base64,${user.avatarImage}`} alt="" />
                    <div className='userConversationInfo'>
                        <span>{user.username}</span>
                    </div>
                </div>
            )}

        </div>
    )
}

export default Searchbar