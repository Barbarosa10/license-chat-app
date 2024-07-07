import React, { useState, useRef, useEffect } from 'react'
import { sentimentAnalysisRoute, sendMessageRoute, updateConversationRoute } from "../utils/ApiRoute";
import { useMessages } from "../context/MessageContext";
import {useConversation} from "../context/ConversationContext";
import {useUser} from "../context/UserContext";
import { useChat } from "../context/ChatContext";
import { useVideoCall } from '../context/VideoCallContext';
import axios from '../utils/axiosConfig';
import { usePopup } from '../context/PopupContext';

const Input = ({socket}) => {
    const [text, setText] = useState("");
    const { currentUser } = useUser();
    const { data } = useChat();
    const { myVideo } = useVideoCall();
    const { addMessage } = useMessages();
    const { updateConversation } = useConversation();
    const { showMessage } = usePopup();
    const canvasRef = useRef();

    useEffect(() => {
        canvasRef.current = document.createElement('canvas');
    }, []);

    const handleSend = async () => {
        try{
            const data_to_send = {
                to: data.user._id,
                from: currentUser._id,
                fromUsername: currentUser.username,
                message: text,
                chatId: data.chatId
            }
    
            socket.current.emit("send-msg", data_to_send);
            const updatedConversation = {
                "id": data.chatId,
                "_id": data.user._id,
                "username": data.user.username,
                "participants": [data.user.username, currentUser.username],
                "lastMessage": text,
                "timestamp": Date.now()
            };
            updateConversation(updatedConversation);

            const message = await axios.post(sendMessageRoute, {
                conversationId: data.chatId,
                sender: currentUser.username,
                message: text,
    
            });
            addMessage(message.data);
            setText("");
    
            if(myVideo.current && canvasRef.current && myVideo.current.getVideoTracks()[0].readyState !== 'ended'){
                const imageCapture = new ImageCapture(myVideo.current.getVideoTracks()[0]);
                const captureFrame = async () => {
                    try {
                      const bitmap = await imageCapture.grabFrame();
              
                      const frameData = await createImageBitmap(bitmap).then((imgBitmap) => {
                        const canvas = document.createElement('canvas');
                        const context = canvas.getContext('2d');

                        const originalWidth = imgBitmap.width;
                        const originalHeight = imgBitmap.height;

                        const newWidth = 640;
                        const newHeight = 360;

                        canvas.width = newWidth;
                        canvas.height = newHeight;
                        context.drawImage(imgBitmap, 0, 0, originalWidth, originalHeight, 0, 0, newWidth, newHeight);
                        return canvas.toDataURL('image/jpeg');
                      });
    
                        if(frameData){
                            const response = await axios.post(sentimentAnalysisRoute, {
                                text: text,
                                videoFrame: frameData
                            });
                            const { image_sentiment, text_sentiment } = response.data;
                            showMessage("Video sentiment: " + image_sentiment + ", Message sentiment: " + text_sentiment);
    
                        }
                        
                    } catch (error) {
                      console.error('Error capturing frame:', error);
                    }
                  };
    
                await captureFrame();
    
            }
            try{
                const conversation = await axios.post(updateConversationRoute, {
                    conversationId: data.chatId,
                    lastMessage: text
                })
            } catch(err){
                console.log(err);
            }
            
        }catch(error){
            console.log(error);
        }

    }

    const handleKeyDown = (e) => {
        if (e.key === 'Enter') {
            handleSend();
        }
    };

    return(
        <div className='input'>
            <input type="text"
            placeholder='Type something...'
            onChange={(e) => setText(e.target.value)}
            onKeyDown={handleKeyDown}
            value={text}
            />
            <div className='send'>
                <button onClick={handleSend}>Send</button>
            </div>
        </div>
    )
}

export default Input