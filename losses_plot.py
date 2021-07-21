import os
import pandas as pd
import matplotlib.pyplot as plt

output_dir = os.path.join('output', 'surf2nude', 'patchnce')
# output_dir = os.path.join('output', 'man2renaissance')
losses_fname = os.path.join(output_dir, 'losses.csv')

# data frame
df = pd.read_csv(losses_fname, index_col=0)

# row by index
# print(df.loc['epoch_0'])
# col by col_name
# print(df['Loss_D'])

print(df)

# x, y
# x = df.index.values
x = range(200)

y_loss_d = df['Loss_D']
y_loss_d_a = df['Loss_D_A']
y_loss_d_b = df['Loss_D_B']

y_loss_g = df['Loss_G']

y_loss_g_identity = df['Loss_G_identity']
y_loss_g_identity_a = df['Loss_G_identity_A']
y_loss_g_identity_b = df['Loss_G_identity_B']

y_loss_g_gan = df['Loss_G_GAN']
y_loss_g_gan_a2b = df['Loss_G_GAN_A2B']
y_loss_g_gan_b2a = df['Loss_G_GAN_B2A']

y_loss_g_cycle = df['Loss_G_cycle']
y_loss_g_cycle_aba = df['Loss_G_cycle_ABA']
y_loss_g_cycle_bab = df['Loss_G_cycle_BAB']

y_loss_g_nce = df['Loss_G_NCE']

# plot
fig = plt.figure(figsize=(12, 6))
ax = plt.subplot(111)

ax.plot(x, y_loss_d, label='Loss_D', color='magenta')
ax.plot(x, y_loss_d_a, label='Loss_D_X')
ax.plot(x, y_loss_d_b, label='Loss_D_Y')

ax.plot(x, y_loss_g, label='Loss_G')

ax.plot(x, y_loss_g_identity, label='Loss_G_identity')
ax.plot(x, y_loss_g_identity_a, label='Loss_G_identity_X')
ax.plot(x, y_loss_g_identity_b, label='Loss_G_identity_Y')

ax.plot(x, y_loss_g_gan, label='Loss_G_GAN')
ax.plot(x, y_loss_g_gan_a2b, label='Loss_G_GAN_X2Y')
ax.plot(x, y_loss_g_gan_b2a, label='Loss_G_GAN_Y2X')

ax.plot(x, y_loss_g_cycle, label='Loss_G_cycle')
ax.plot(x, y_loss_g_cycle_aba, label='Loss_G_cycle_XYX')
ax.plot(x, y_loss_g_cycle_bab, label='Loss_G_cycle_YXY')

ax.plot(x, y_loss_g_nce, label='Loss_G_NCE')

ax.legend(loc='upper left', bbox_to_anchor=(1, 1.02))

plt.xlabel('Epoch')
plt.ylabel('Loss')

fig.savefig('loss.png', bbox_inches='tight', dpi=fig.dpi)
# plt.show()