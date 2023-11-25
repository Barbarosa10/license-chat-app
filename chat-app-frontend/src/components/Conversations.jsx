import React, { useEffect, useState, useRef } from 'react'

const Conversations = ({conversations}) => {
    // console.log(conversations);

    return(

        <div className='conversations'>
            {conversations.map((conversation, index) => {
                return (
                    <div key={index} className='userConversation'>
                        <img src={`data:image/;base64,${conversation.avatarImage}`} alt="" />
                        <div className='userConversationInfo'>
                            <span>{conversation.username}</span>
                            <p>{conversation.message}</p>
                        </div>
                    </div>
                )
            })};
        </div>
            //  {/* <div className='userConversation'>
            // //     <img src={`data:image/svg+xml;base64,${contact.avatarImage}`} alt="" />
            // //     <div className='userConversationInfo'>
            // //         <span>{contact.username}</span>
            // //         <p>Hello world!</p>
            // //     </div>
            // // </div>

        
    )
}

export default Conversations