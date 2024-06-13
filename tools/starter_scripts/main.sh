# 删除临时文件 clean
rm -f ./temp_* && rm -f ./tmp/*

# 把上一次的tmux窗口删除
tmux kill-session -t $tmux_s_name
tmux kill-session -t fake
# 得先制造一个无用的session，否则无法通过conf文件新建session。不太清楚为什么，玄学。
tmux new-session -d -s fake

# 创建一个新的tmux会话，并命名为my_session
tmux new-session -d -s $tmux_s_name
# 加载配置文件定制当前tmux会话
tmux source-file tools/starter_scripts/main.tmux.conf
echo 'Create Tmux!!'

echo 'sleep 3s, wait zsh init.Maybe nothing.'
sleep 3

# # 将命令分别发送到每个窗格并在前台执行
split -l 1 -d $commands_file temp_
mv -f ./temp_* ./tmp

# sendkey_w1
for i in 0 1 2 3; do
    # 这一步是避免zsh的更新提示
    tmux send-keys -t $tmux_s_name:w1.$i "n" Enter
    # 先切到对应的conda环境
    tmux send-keys -t $tmux_s_name:w1.$i "conda activate LLML" Enter
    sleep 1
    tmux send-keys -t $tmux_s_name:w1.$i "cat ./tmp/temp_0$i | xargs -I {} sh -c 'echo {};{}'" Enter
done

# sendkey_w2
for i in 4 5 6 7; do
    # 这一步是避免zsh的更新提示
    tmux send-keys -t $tmux_s_name:w2.$(($i - 4)) "n" Enter
    tmux send-keys -t $tmux_s_name:w2.$(($i - 4)) "conda activate LLML" Enter
    sleep 1
    tmux send-keys -t $tmux_s_name:w2.$(($i - 4)) "cat ./tmp/temp_0$i | xargs -I {} sh -c 'echo {};{}'" Enter
done

