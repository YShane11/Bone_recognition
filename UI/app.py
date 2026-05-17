import streamlit as st
import base64


def main():
    st.title("Bone Recognize")
    st.subheader("鏡頭畫面")

    # 上傳影片
    uploaded_file = st.file_uploader("請上傳影片", type=["mp4", "avi", "mov"])

    st.video(uploaded_file)


if __name__ == '__main__':
    main() # streamlit run UI/app.py
    